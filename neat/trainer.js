// Обучение динозавров через NEAT на копии игры из Chromium (../t-rex-runner).
// Все динозавры поколения бегут по одной трассе: те же кактусы и птицы,
// та же скорость. Каждым управляет своя нейросеть. Приспособленность —
// пройденное расстояние. Когда погибают все, NEAT собирает новое поколение.
(async function () {
  'use strict';

  const STEP_MS = 1000 / 60; // шаг симуляции — один кадр при 60 Гц
  const INPUT_NAMES = ['до препятствия', 'ширина', 'высота', 'над землёй', 'скорость', 'высота дино', 'скорость дино', 'зазор', 'смещение'];
  const OUTPUT_NAMES = ['прыжок', 'пригнуться'];

  // Классы игры спрятаны внутри её модуля. Подгружаем исходник и
  // выставляем нужные классы наружу, сам файл игры не меняем.
  async function loadGame() {
    let src = await (await fetch('../t-rex-runner/index.js')).text();
    const patches = [
      ["window['Runner'] = Runner;", "window['Runner'] = Runner; window.DinoGame = { Trex: Trex, Horizon: Horizon, DistanceMeter: DistanceMeter, checkForCollision: checkForCollision };"],
      ["document.addEventListener('DOMContentLoaded', onDocumentLoad);", ''],
    ];
    for (const [from, to] of patches) {
      if (!src.includes(from)) throw new Error('Не нашёл в коде игры: ' + from);
      src = src.replace(from, to);
    }
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
  }

  await loadGame();
  const { Trex, Horizon, DistanceMeter, checkForCollision } = window.DinoGame;
  const Runner = window.Runner;
  const RC = Runner.config;
  const DIM = Runner.defaultDimensions;

  const hidpi = window.devicePixelRatio > 1;
  Runner.imageSprite = document.getElementById(hidpi ? 'offline-resources-2x' : 'offline-resources-1x');
  const spriteDef = hidpi ? Runner.spriteDefinition.HDPI : Runner.spriteDefinition.LDPI;
  if (!Runner.imageSprite.complete) await new Promise((ok) => Runner.imageSprite.addEventListener('load', ok));

  const canvas = document.createElement('canvas');
  canvas.width = DIM.WIDTH;
  canvas.height = DIM.HEIGHT;
  document.getElementById('game').appendChild(canvas);
  Runner.updateCanvasScaling(canvas);
  const ctx = canvas.getContext('2d');

  // Игра рисует прямо во время обновления. Пока идут шаги симуляции,
  // рисование выключено; картинка строится один раз за кадр экрана.
  let drawing = false;
  const realDrawImage = CanvasRenderingContext2D.prototype.drawImage;
  ctx.drawImage = function () {
    if (drawing) realDrawImage.apply(this, arguments);
  };

  const ui = {
    gen: document.getElementById('stat-gen'),
    track: document.getElementById('stat-track'),
    alive: document.getElementById('stat-alive'),
    species: document.getElementById('stat-species'),
    score: document.getElementById('stat-score'),
    record: document.getElementById('stat-record'),
    mode: document.getElementById('mode'),
    net: document.getElementById('net'),
    netCaption: document.getElementById('net-caption'),
    chart: document.getElementById('chart'),
    pop: document.getElementById('pop-size'),
    cap: document.getElementById('score-cap'),
    tracks: document.getElementById('tracks'),
    champion: document.getElementById('btn-champion'),
  };

  let pop = null;
  let speed = 1; // шагов симуляции за кадр экрана; Infinity — сколько успеем
  let showChampion = false;
  let championRuns = []; // очки лучшего в забегах, пока его показываем
  let highScore = 0; // лучший забег вообще, для «HI» в игре
  let paused = false;
  // Каждое поколение проходит несколько трасс подряд: одна удачная трасса
  // не делает сеть чемпионом, считается средний результат.
  const gen = { track: 0, tracks: 1 };
  const run = {};
  let meter = null;

  const points = (distance) => meter.getActualDistance(Math.ceil(distance));

  function startRun(genomes) {
    drawing = false;
    run.horizon = new Horizon(canvas, spriteDef, DIM, RC.GAP_COEFFICIENT);
    run.speed = RC.SPEED;
    run.distance = 0;
    run.time = 0;
    run.over = false;
    run.dinos = genomes.map((genome) => {
      const t = new Trex(canvas, spriteDef.TREX);
      t.xPos = Trex.config.START_X_POS;
      t.update(0, Trex.status.RUNNING);
      return { t, genome, alive: true, distance: 0 };
    });
    run.alive = run.dinos.length;
    meter.reset();
  }

  // Что видит сеть. Все значения примерно в пределах 0…1.
  function senseObstacles() {
    const x = Trex.config.START_X_POS;
    const ahead = run.horizon.obstacles.filter((o) => o.xPos + o.width > x);
    const o = ahead[0];
    const o2 = ahead[1];
    if (!o) return { dist: 1, width: 0, height: 0, lift: 0, gap: 1 };
    return {
      dist: Math.max(-0.2, (o.xPos - x - Trex.config.WIDTH) / DIM.WIDTH),
      width: o.width / 75,
      height: o.typeConfig.height / 50,
      // Насколько низ препятствия поднят над землёй: у кактусов 0, у птиц больше.
      lift: (DIM.HEIGHT - RC.BOTTOM_PAD - o.yPos - o.typeConfig.height) / 100,
      gap: o2 ? (o2.xPos - o.xPos - o.width) / DIM.WIDTH : 1,
    };
  }

  function act(d, s) {
    const t = d.t;
    const out = d.genome.activate([
      s.dist,
      s.width,
      s.height,
      s.lift,
      (run.speed - RC.SPEED) / (RC.MAX_SPEED - RC.SPEED),
      (t.groundYPos - t.yPos) / 100,
      t.jumping ? t.jumpVelocity / 10 : 0,
      s.gap,
    ]);
    const wantJump = out[0] > 0.5;
    const wantDuck = out[1] > 0.5;
    // Как клавиши: в воздухе «вниз» ускоряет падение, на земле — пригибает.
    if (t.jumping) {
      if (wantDuck && !t.speedDrop) t.setSpeedDrop();
    } else if (wantJump) {
      if (t.ducking) t.setDuck(false);
      t.startJump(run.speed);
    } else if (wantDuck !== t.ducking) {
      t.setDuck(wantDuck);
    }
  }

  // Один кадр игры — тот же порядок, что в Runner.update.
  function step() {
    run.time += STEP_MS;
    const hasObstacles = run.time > RC.CLEAR_TIME;
    const sense = senseObstacles();
    for (const d of run.dinos) if (d.alive) act(d, sense);
    for (const d of run.dinos) if (d.alive && d.t.jumping) d.t.updateJump(STEP_MS);
    run.horizon.update(STEP_MS, run.speed, hasObstacles, false);

    // Игра проверяет столкновение только с первым препятствием в списке.
    const first = run.horizon.obstacles[0];
    if (hasObstacles && first) {
      for (const d of run.dinos) {
        if (d.alive && checkForCollision(first, d.t)) {
          d.alive = false;
          d.distance = run.distance;
          run.alive--;
        }
      }
    }
    if (!run.alive) {
      run.over = true;
      return;
    }

    run.distance += run.speed;
    if (run.speed < RC.MAX_SPEED) run.speed += RC.ACCELERATION;
    meter.update(STEP_MS, Math.ceil(run.distance));
    for (const d of run.dinos) if (d.alive) d.t.update(STEP_MS);

    const cap = Number(ui.cap.value);
    if (!showChampion && cap > 0 && points(run.distance) >= cap) {
      for (const d of run.dinos) if (d.alive) d.distance = run.distance;
      run.over = true;
    }
  }

  function render() {
    drawing = true;
    ctx.clearRect(0, 0, DIM.WIDTH, DIM.HEIGHT);
    run.horizon.update(0, run.speed, run.time > RC.CLEAR_TIME, false);
    meter.update(0, Math.ceil(run.distance));
    const leader = leaderDino();
    for (const d of run.dinos) {
      if (!d.alive || d === leader) continue;
      ctx.globalAlpha = 0.25;
      d.t.update(0);
    }
    ctx.globalAlpha = 1;
    if (leader) leader.t.update(0);
    drawing = false;
  }

  // Первый в поколении — чемпион прошлого; пока он жив, показываем его.
  function leaderDino() {
    return run.dinos.find((d) => d.alive) || null;
  }

  // endGeneration — не проходить оставшиеся трассы (кнопка «Следующее поколение»).
  function finishRun(endGeneration) {
    highScore = Math.max(highScore, run.distance);
    meter.setHighScore(Math.ceil(highScore));
    if (showChampion) {
      championRuns.push(points(run.distance));
      startRun([pop.best.clone()]);
      return;
    }
    for (const d of run.dinos) d.genome.total += d.distance;
    gen.track++;
    if (gen.track < gen.tracks && !endGeneration) {
      startRun(pop.genomes);
      return;
    }
    for (const g of pop.genomes) g.fitness = g.total / gen.track;
    pop.evolve();
    drawChart();
    startGeneration();
  }

  function startGeneration() {
    gen.track = 0;
    gen.tracks = Math.max(1, Math.min(10, Number(ui.tracks.value) || 1));
    for (const g of pop.genomes) g.total = 0;
    startRun(pop.genomes);
  }

  function newPopulation() {
    pop = new NEAT.Population({
      popSize: Math.max(10, Math.min(500, Number(ui.pop.value) || 150)),
      inputs: INPUT_NAMES.length - 1,
      outputs: OUTPUT_NAMES.length,
    });
    window.dinoNeat = pop; // для отладки из консоли
    showChampion = false;
    ui.champion.textContent = 'Показать лучшего';
    highScore = 0;
    meter.setHighScore(0);
    startGeneration();
    drawChart();
  }

  // Симуляция идёт по реальному времени на своём таймере, отдельно от
  // отрисовки: так скорость не зависит от частоты монитора, а обучение
  // не замирает, когда вкладка в фоне. На «Макс» шаги идут без пауз,
  // с короткими перерывами, чтобы страница отвечала.
  const channel = new MessageChannel();
  channel.port1.onmessage = (e) => tick(e.data);
  let lastTick = 0;
  let backlog = 0;
  let loop = 0; // номер текущего цикла: старый цикл при перезапуске затихает

  function restartLoop() {
    loop++;
    lastTick = performance.now();
    backlog = 0;
    schedule(loop);
  }

  function schedule(id) {
    if (speed === Infinity) channel.port2.postMessage(id);
    else setTimeout(tick, 4, id);
  }

  function advance() {
    step();
    if (run.over) finishRun();
  }

  function tick(id) {
    if (id !== loop) return;
    const now = performance.now();
    const elapsed = Math.min(250, now - lastTick);
    lastTick = now;
    if (!paused) {
      if (speed === Infinity) {
        while (performance.now() < now + 12) advance();
      } else {
        backlog += elapsed * speed;
        while (backlog >= STEP_MS) {
          backlog -= STEP_MS;
          advance();
        }
      }
    }
    schedule(id);
  }

  function frame() {
    requestAnimationFrame(frame);
    render();
    updateStats();
    drawNet();
  }

  function updateStats() {
    ui.gen.textContent = pop.generation;
    ui.track.textContent = showChampion ? '—' : gen.track + 1 + ' из ' + gen.tracks;
    ui.alive.textContent = run.alive + ' из ' + run.dinos.length;
    ui.species.textContent = pop.species.length;
    ui.score.textContent = points(run.distance);
    ui.record.textContent = pop.best ? points(pop.best.fitness) + ' (пок. ' + pop.best.generation + ')' : '—';
    if (!showChampion) ui.mode.textContent = 'Обучение';
    else if (!championRuns.length) ui.mode.textContent = 'Бежит лучший за всё время';
    else ui.mode.textContent = 'Бежит лучший за всё время. Прошлые забеги: ' + championRuns.slice(-5).join(', ');
  }

  // --- Рисунок сети ведущего динозавра ---

  function themeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function fitCanvas(c) {
    const r = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) {
      c.width = Math.round(w * r);
      c.height = Math.round(h * r);
    }
    const g = c.getContext('2d');
    g.setTransform(r, 0, 0, r, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  }

  function drawNet() {
    const leader = leaderDino();
    if (!leader) return;
    const genome = leader.genome;
    const net = genome.net || genome.build();
    const { g, w, h } = fitCanvas(ui.net);
    const { nIn, nOut } = pop;

    // Глубина скрытого нейрона — самый длинный путь до него от входов.
    const depth = new Map();
    for (let i = 0; i < nIn; i++) depth.set(i, 0);
    for (const id of net.order) {
      let d = 0;
      for (const c of net.incoming.get(id)) d = Math.max(d, (depth.get(c.from) || 0) + 1);
      depth.set(id, d);
    }
    const maxHidden = Math.max(0, ...genome.hidden.map((id) => depth.get(id) || 1));
    const columns = maxHidden + 1;

    const left = 108;
    const right = w - 86;
    const pos = new Map();
    const place = (ids, col) => {
      ids.forEach((id, k) => {
        pos.set(id, {
          x: left + ((right - left) * col) / columns,
          y: ((k + 1) * h) / (ids.length + 1),
        });
      });
    };
    place([...Array(nIn).keys()], 0);
    place(Array.from({ length: nOut }, (_, i) => nIn + i), columns);
    for (let col = 1; col <= maxHidden; col++) {
      place(genome.hidden.filter((id) => (depth.get(id) || 1) === col), col);
    }

    const pos1 = themeColor('--pos');
    const neg1 = themeColor('--neg');
    const ink = themeColor('--ink');
    const muted = themeColor('--muted');
    const v = net.values;

    for (const c of genome.conns) {
      if (!c.enabled) continue;
      const a = pos.get(c.from);
      const b = pos.get(c.to);
      if (!a || !b) continue;
      const signal = Math.min(1, Math.abs((v.get(c.from) || 0) * c.weight));
      g.globalAlpha = 0.25 + 0.75 * signal;
      g.strokeStyle = c.weight > 0 ? pos1 : neg1;
      g.lineWidth = 0.5 + Math.min(3.5, Math.abs(c.weight));
      g.beginPath();
      g.moveTo(a.x, a.y);
      g.lineTo(b.x, b.y);
      g.stroke();
    }
    g.globalAlpha = 1;

    g.font = '12px system-ui, sans-serif';
    g.textBaseline = 'middle';
    for (const [id, p] of pos) {
      const value = v.get(id) || 0;
      g.beginPath();
      g.arc(p.x, p.y, 6, 0, Math.PI * 2);
      g.fillStyle = muted;
      g.fill();
      g.beginPath();
      g.arc(p.x, p.y, 6 * Math.min(1, Math.abs(value)), 0, Math.PI * 2);
      g.fillStyle = ink;
      g.fill();
      g.fillStyle = ink;
      if (id < nIn) {
        g.textAlign = 'right';
        g.fillText(INPUT_NAMES[id], p.x - 11, p.y);
      } else if (id < nIn + nOut) {
        g.textAlign = 'left';
        const on = value > 0.5;
        g.font = (on ? '600 ' : '') + '12px system-ui, sans-serif';
        g.fillText(OUTPUT_NAMES[id - nIn], p.x + 11, p.y);
        g.font = '12px system-ui, sans-serif';
      }
    }
    const links = genome.conns.filter((c) => c.enabled).length;
    ui.netCaption.textContent =
      'Нейронов в скрытом слое: ' + genome.hidden.length + ', связей: ' + links + '. Вид №' + (genome.species ? genome.species.id : '—') + '.';
  }

  // --- График по поколениям ---

  function drawChart() {
    const { g, w, h } = fitCanvas(ui.chart);
    const hist = pop.history;
    const ink = themeColor('--ink');
    const muted = themeColor('--muted');
    const best = themeColor('--series-1');
    const mean = themeColor('--series-2');
    g.font = '12px system-ui, sans-serif';
    if (!hist.length) {
      g.fillStyle = muted;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText('График появится после первого поколения', w / 2, h / 2);
      return;
    }
    const pad = { l: 44, r: 12, t: 12, b: 24 };
    const maxY = Math.max(10, ...hist.map((p) => points(p.best)));
    const n = hist.length;
    const X = (i) => pad.l + (n === 1 ? 0 : ((w - pad.l - pad.r) * i) / (n - 1));
    const Y = (val) => h - pad.b - ((h - pad.t - pad.b) * val) / maxY;

    g.strokeStyle = muted;
    g.globalAlpha = 0.35;
    g.lineWidth = 1;
    g.fillStyle = muted;
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const val = (maxY * k) / 4;
      g.beginPath();
      g.moveTo(pad.l, Y(val));
      g.lineTo(w - pad.r, Y(val));
      g.stroke();
    }
    g.globalAlpha = 1;
    for (let k = 0; k <= 4; k++) {
      const val = (maxY * k) / 4;
      g.fillText(Math.round(val), pad.l - 6, Y(val));
    }
    g.textAlign = 'center';
    g.textBaseline = 'top';
    g.fillText('поколение ' + hist[n - 1].generation, w - pad.r - 50, h - pad.b + 6);

    const line = (key, color) => {
      g.strokeStyle = color;
      g.lineWidth = 2;
      g.beginPath();
      hist.forEach((p, i) => {
        const y = Y(points(p[key]));
        if (i) g.lineTo(X(i), y);
        else g.moveTo(X(i), y);
      });
      g.stroke();
      if (n === 1) {
        g.fillStyle = color;
        g.beginPath();
        g.arc(X(0), Y(points(hist[0][key])), 3, 0, Math.PI * 2);
        g.fill();
      }
    };
    line('mean', mean);
    line('best', best);
    g.fillStyle = ink;
  }

  // --- Кнопки ---

  document.querySelectorAll('[data-speed]').forEach((b) => {
    b.addEventListener('click', () => {
      speed = b.dataset.speed === 'max' ? Infinity : Number(b.dataset.speed);
      restartLoop();
      document.querySelectorAll('[data-speed]').forEach((x) => x.setAttribute('aria-pressed', x === b));
    });
  });
  document.getElementById('btn-pause').addEventListener('click', (e) => {
    paused = !paused;
    e.currentTarget.textContent = paused ? 'Продолжить' : 'Пауза';
  });
  document.getElementById('btn-reset').addEventListener('click', newPopulation);
  document.getElementById('btn-skip').addEventListener('click', () => {
    if (showChampion) return;
    for (const d of run.dinos) if (d.alive) d.distance = run.distance;
    finishRun(true);
  });
  ui.champion.addEventListener('click', () => {
    if (!pop.best && !showChampion) return;
    highScore = Math.max(highScore, run.distance);
    meter.setHighScore(Math.ceil(highScore));
    showChampion = !showChampion;
    championRuns = [];
    window.dinoChampionRuns = championRuns; // для отладки из консоли
    ui.champion.textContent = showChampion ? 'Вернуться к обучению' : 'Показать лучшего';
    startRun(showChampion ? [pop.best.clone()] : pop.genomes);
  });
  window.addEventListener('resize', drawChart);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawChart);

  meter = new DistanceMeter(canvas, spriteDef.TEXT_SPRITE, DIM.WIDTH);
  newPopulation();
  restartLoop();
  requestAnimationFrame(frame);
})();
