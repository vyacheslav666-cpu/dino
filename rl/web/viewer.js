// Обученная Q-сеть играет в настоящую игру в браузере. Картинку для сети
// готовит тот же код, что и при обучении (../game.js), сеть считается
// через TensorFlow.js по весам из model/ (их выгружает rl/export.py).
(async function () {
  'use strict';

  const { patchSource, makeRouter, Game, OBS, SKIP, STEP_MS, setupObsContext } = DinoRL;
  const ACTION_NAMES = ['ничего', 'прыжок', 'вниз'];
  const DECISION_MS = STEP_MS * SKIP;
  const FRAME = OBS.height * OBS.width;

  const ui = {
    status: document.getElementById('status'),
    score: document.getElementById('stat-score'),
    best: document.getElementById('stat-best'),
    runs: document.getElementById('stat-runs'),
    speed: document.getElementById('stat-speed'),
    ms: document.getElementById('stat-ms'),
    q: document.getElementById('q'),
    frames: document.getElementById('frames'),
    history: document.getElementById('runs'),
    pause: document.getElementById('btn-pause'),
  };

  async function loadGame() {
    let src = patchSource(await (await fetch('../../t-rex-runner/index.js')).text());
    // Сеть училась на спрайтах обычной плотности; на экранах HiDPI игра
    // иначе взяла бы двойные координаты в спрайте.
    const hidpi = 'var IS_HIDPI = window.devicePixelRatio > 1;';
    if (!src.includes(hidpi)) throw new Error('Не нашёл в коде игры: ' + hidpi);
    src = src.replace(hidpi, 'var IS_HIDPI = false;');
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
    const sprite = document.getElementById('sprite');
    if (!sprite.complete) await new Promise((ok) => sprite.addEventListener('load', ok));
    window.Runner.imageSprite = sprite;
    return { Runner: window.Runner, ...window.DinoGame, spriteDef: window.Runner.spriteDefinition.LDPI };
  }

  async function loadModel() {
    const meta = await (await fetch('model/model.json')).json();
    const data = new Float32Array(await (await fetch('model/weights.bin')).arrayBuffer());
    const w = {};
    for (const l of meta.layers) {
      const size = l.shape.reduce((a, b) => a * b, 1);
      w[l.name] = tf.tensor(data.subarray(l.offset, l.offset + size), l.shape);
    }
    return { w, source: meta.source };
  }

  // То же, что QNet.forward в rl/model.py. Вход — стопка кадров [4, H, W].
  function makeForward(w) {
    const convs = [[0, 3], [2, 2], [4, 1]]; // индекс слоя в nn.Sequential, шаг
    return function (stack) {
      return tf.tidy(() => {
        let h = tf.tensor(stack, [OBS.stack, OBS.height, OBS.width], 'float32').transpose([1, 2, 0]).expandDims(0).div(255);
        for (const [i, stride] of convs) {
          h = tf.relu(tf.conv2d(h, w[`conv.${i}.weight`], stride, 'valid').add(w[`conv.${i}.bias`]));
        }
        // PyTorch раскладывает признаки как [каналы, высота, ширина].
        const f = h.transpose([0, 3, 1, 2]).reshape([1, -1]);
        const head = (name) => tf.relu(f.matMul(w[`${name}.0.weight`]).add(w[`${name}.0.bias`]))
          .matMul(w[`${name}.2.weight`]).add(w[`${name}.2.bias`]);
        const v = head('value');
        const a = head('advantage');
        return v.add(a).sub(a.mean(1, true)).dataSync();
      });
    };
  }

  // Холсты
  const view = document.getElementById('view');
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  view.width = 600 * dpr;
  view.height = 150 * dpr;
  const viewCtx = view.getContext('2d');

  const obsCanvas = document.createElement('canvas');
  obsCanvas.width = 600;
  obsCanvas.height = 150;
  const obsCtx = setupObsContext(obsCanvas.getContext('2d', { willReadFrequently: true }));

  const frameViews = ['3 хода назад', '2 хода назад', '1 ход назад', 'сейчас'].map((label) => {
    const fig = document.createElement('figure');
    const c = document.createElement('canvas');
    c.width = OBS.width;
    c.height = OBS.height;
    const cap = document.createElement('figcaption');
    cap.textContent = label;
    fig.append(c, cap);
    ui.frames.appendChild(fig);
    return { ctx: c.getContext('2d'), img: new ImageData(OBS.width, OBS.height) };
  });

  const qRows = ACTION_NAMES.map((name) => {
    const label = document.createElement('div');
    label.textContent = name;
    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('div');
    fill.className = 'fill';
    track.appendChild(fill);
    const val = document.createElement('div');
    val.className = 'val';
    ui.q.append(label, track, val);
    return { label, track, fill, val };
  });

  let lib, game, forward;
  try {
    lib = await loadGame();
    const model = await loadModel();
    forward = makeForward(model.w);
    ui.status.textContent = 'Веса: ' + model.source + '.';
  } catch (err) {
    console.error(err);
    ui.status.textContent = 'Нет весов сети. Обучите её (rl/train.py) и выгрузите (rl/export.py).';
    return;
  }
  game = new Game(lib, makeRouter());

  const stack = new Uint8Array(OBS.stack * FRAME);
  let q = [0, 0, 0];
  let action = 0;
  let inferMs = 0;
  let speed = 1;
  let paused = false;
  let crashPause = 0;
  let best = 0;
  const history = [];

  function newRun() {
    game.reset();
    game.observe(obsCtx, stack, (OBS.stack - 1) * FRAME);
    for (let k = 0; k < OBS.stack - 1; k++) stack.copyWithin(k * FRAME, (OBS.stack - 1) * FRAME, OBS.stack * FRAME);
    q = [0, 0, 0];
  }

  function decide() {
    const t0 = performance.now();
    q = Array.from(forward(stack));
    inferMs = inferMs * 0.9 + (performance.now() - t0) * 0.1;
    action = q.indexOf(Math.max(...q));
    const crashed = game.step(action);
    stack.copyWithin(0, FRAME);
    game.observe(obsCtx, stack, (OBS.stack - 1) * FRAME);
    return crashed;
  }

  function finishRun() {
    const score = game.score();
    best = Math.max(best, score);
    history.unshift(score);
    if (history.length > 12) history.pop();
    crashPause = 1200;
  }

  function render() {
    viewCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    viewCtx.imageSmoothingEnabled = false;
    viewCtx.fillStyle = '#f7f7f7';
    viewCtx.fillRect(0, 0, 600, 150);
    game.draw(viewCtx);
    if (game.crashed) {
      viewCtx.fillStyle = '#535353';
      viewCtx.font = 'bold 16px monospace';
      viewCtx.textAlign = 'center';
      viewCtx.fillText('G A M E   O V E R', 300, 60);
    }

    for (let k = 0; k < OBS.stack; k++) {
      const { ctx, img } = frameViews[k];
      for (let i = 0; i < FRAME; i++) {
        const v = stack[k * FRAME + i];
        img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }

    const lo = Math.min(...q, 0);
    const hi = Math.max(...q, 1e-6);
    qRows.forEach((row, i) => {
      const chosen = i === action && !game.crashed;
      row.label.className = chosen ? 'chosen' : '';
      row.track.className = 'track' + (chosen ? ' chosen' : '');
      row.fill.style.width = (100 * (q[i] - lo) / (hi - lo || 1)).toFixed(1) + '%';
      row.val.textContent = q[i].toFixed(2);
    });

    ui.score.textContent = String(game.score()).padStart(5, '0');
    ui.best.textContent = best ? String(best).padStart(5, '0') : '—';
    ui.runs.textContent = history.length ? String(history.length) : '0';
    ui.speed.textContent = game.speed.toFixed(1);
    ui.ms.textContent = inferMs ? inferMs.toFixed(1) + ' мс' : '—';
    ui.history.textContent = history.length ? 'Последние забеги: ' + history.join(', ') : '';
  }

  let backlog = 0;
  let last = performance.now();
  function tick(now) {
    const elapsed = Math.min(now - last, 100);
    last = now;
    if (!paused) {
      if (crashPause > 0) {
        crashPause -= elapsed;
        if (crashPause <= 0) newRun();
      } else {
        backlog += elapsed * speed;
        // Если сеть не успевает, игра замедляется, а не пропускает решения.
        for (let n = 0; backlog >= DECISION_MS && n < 12; n++) {
          backlog -= DECISION_MS;
          if (decide()) {
            finishRun();
            backlog = 0;
            break;
          }
        }
        backlog = Math.min(backlog, DECISION_MS * 2);
      }
    }
    render();
    requestAnimationFrame(tick);
  }

  document.querySelectorAll('[data-speed]').forEach((b) => {
    b.addEventListener('click', () => {
      speed = Number(b.dataset.speed);
      document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', x === b));
    });
  });
  ui.pause.addEventListener('click', () => {
    paused = !paused;
    ui.pause.textContent = paused ? 'Продолжить' : 'Пауза';
  });
  document.getElementById('btn-restart').addEventListener('click', () => {
    crashPause = 0;
    newRun();
  });

  // Для отладки из консоли: сверка с PyTorch на той же стопке кадров.
  window.dinoViewer = { stack, forward, game };

  ui.runs.textContent = '0';
  newRun();
  requestAnimationFrame(tick);
})();
