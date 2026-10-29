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
  // Если передан acts, туда же кладутся активации слоёв для схемы мозга.
  function makeForward(w) {
    const convs = [[0, 3], [2, 2], [4, 1]]; // индекс слоя в nn.Sequential, шаг
    return function (stack, acts) {
      return tf.tidy(() => {
        let h = tf.tensor(stack, [OBS.stack, OBS.height, OBS.width], 'float32').transpose([1, 2, 0]).expandDims(0).div(255);
        convs.forEach(([i, stride], k) => {
          h = tf.relu(tf.conv2d(h, w[`conv.${i}.weight`], stride, 'valid').add(w[`conv.${i}.bias`]));
          if (acts) acts['c' + (k + 1)] = h.dataSync();
        });
        // PyTorch раскладывает признаки как [каналы, высота, ширина].
        const f = h.transpose([0, 3, 1, 2]).reshape([1, -1]);
        const hidden = (name) => tf.relu(f.matMul(w[`${name}.0.weight`]).add(w[`${name}.0.bias`]));
        const hv = hidden('value');
        const ha = hidden('advantage');
        const v = hv.matMul(w['value.2.weight']).add(w['value.2.bias']);
        const a = ha.matMul(w['advantage.2.weight']).add(w['advantage.2.bias']);
        if (acts) {
          acts.input = stack.slice();
          acts.hv = hv.dataSync();
          acts.ha = ha.dataSync();
          acts.v = v.dataSync()[0];
        }
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

  let lib, game, forward, brain;
  try {
    lib = await loadGame();
    const model = await loadModel();
    forward = makeForward(model.w);
    brain = DinoBrain.create(document.getElementById('brain'), {
      value: model.w['value.2.weight'].dataSync(),
      advantage: model.w['advantage.2.weight'].dataSync(),
    });
    ui.status.textContent = '';
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
  let paused = false;
  let crashPause = 0;
  let best = 0;
  let runs = 0;
  let acts = {};
  let brainDirty = true;

  function newRun() {
    game.reset();
    game.observe(obsCtx, stack, (OBS.stack - 1) * FRAME);
    for (let k = 0; k < OBS.stack - 1; k++) stack.copyWithin(k * FRAME, (OBS.stack - 1) * FRAME, OBS.stack * FRAME);
    q = [0, 0, 0];
  }

  // Активации читаются из видеокарты только для хода, который попадёт на экран.
  function decide(show) {
    const t0 = performance.now();
    if (show) {
      acts = {};
      brainDirty = true;
    }
    q = Array.from(forward(stack, show ? acts : null));
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
    runs++;
    crashPause = 1200;
  }

  // Табло поверх игры, только на экране: сеть получает кадры с отдельного
  // холста и его не видит. Очки — цифрами из спрайта игры, как в оригинале.
  const DIGIT = { w: 10, h: 13, step: 11 };
  function drawDigits(text, x, y) {
    const sprite = window.Runner.imageSprite;
    const base = lib.spriteDef.TEXT_SPRITE;
    for (const ch of text) {
      // В спрайте после цифр 0–9 идут буквы H и I.
      const i = ch === 'H' ? 10 : ch === 'I' ? 11 : ch === ' ' ? -1 : Number(ch);
      if (i >= 0) viewCtx.drawImage(sprite, base.x + i * DIGIT.w, base.y, DIGIT.w, DIGIT.h, x, y, DIGIT.w, DIGIT.h);
      x += DIGIT.step;
    }
  }

  function drawHud() {
    const score = String(game.score()).padStart(5, '0');
    const hi = 'HI ' + String(best).padStart(5, '0') + '  ';
    const x = 600 - DIGIT.step * (score.length + 1);
    if (best) {
      viewCtx.globalAlpha = 0.8;
      drawDigits(hi, x - DIGIT.step * hi.length, 5);
      viewCtx.globalAlpha = 1;
    }
    drawDigits(score, x, 5);

    viewCtx.fillStyle = '#535353';
    viewCtx.font = '10px "Press Start 2P", monospace';
    viewCtx.textAlign = 'left';
    viewCtx.textBaseline = 'top';
    const parts = [
      'ЗАБЕГОВ ' + runs,
      'СКОРОСТЬ ' + game.speed.toFixed(1),
      'СЕТЬ ' + (inferMs ? inferMs.toFixed(0) + ' МС' : '—'),
    ];
    viewCtx.fillText(parts.join('   '), 8, 8);
    if (paused) {
      viewCtx.textAlign = 'center';
      viewCtx.fillText('ПАУЗА', 300, 60);
    }
    viewCtx.textBaseline = 'alphabetic';
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
    drawHud();

    if (brainDirty) {
      brain.draw(acts, q, action, ACTION_NAMES);
      brainDirty = false;
    }
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
        backlog += elapsed;
        // Если сеть не успевает, игра замедляется, а не пропускает решения.
        for (let n = 0; backlog >= DECISION_MS && n < 12; n++) {
          backlog -= DECISION_MS;
          if (decide(backlog < DECISION_MS)) {
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

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { brainDirty = true; });
  // Пробел — пауза, чтобы разглядеть схему мозга на одном ходу.
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.repeat) return;
    e.preventDefault();
    paused = !paused;
  });

  // Для отладки из консоли: сверка с PyTorch на той же стопке кадров.
  window.dinoViewer = { stack, forward, game };

  newRun();
  requestAnimationFrame(tick);
})();
