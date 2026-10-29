// Среда для обучения с подкреплением: несколько партий настоящей игры
// из ../t-rex-runner без браузера. Агент видит только картинку.
//
// Протокол через stdin/stdout, всё двоичное.
//   При запуске пишем:   кадры всех партий.
//   Читаем:              'S' + N байт действий (0 — ничего, 1 — пробел, 2 — вниз).
//   Отвечаем:            N × (done u8, truncated u8, score f32le), потом N кадров.
//   Читаем 'Q' — выходим.
// Кадр — 40 × 150 байт, оттенки серого, фон 0, рисунок светлый.
// Если партия кончилась, вместо её кадра приходит первый кадр новой партии.
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const { patchSource, makeRouter, Game, OBS, setupObsContext } = require('./game');

function parseArgs() {
  const args = { envs: 8, seed: 1, maxScore: 0 };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '').replace(/-(\w)/g, (_, c) => c.toUpperCase());
    if (!(key in args)) throw new Error('Неизвестный параметр ' + argv[i]);
    args[key] = Number(argv[i + 1]);
  }
  return args;
}

// Детерминированный генератор вместо Math.random: при одном seed трассы
// повторяются.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Исходник игры запускаем в отдельном контексте с заглушками браузера.
async function loadGame(seed) {
  const gameDir = path.join(__dirname, '..', 't-rex-runner');
  const src = patchSource(fs.readFileSync(path.join(gameDir, 'index.js'), 'utf8'));
  const window = { devicePixelRatio: 1, navigator: { platform: '', userAgent: '' } };
  const sandbox = {
    window,
    document: { addEventListener() {} },
    performance: { now: () => 0 },
    __random: mulberry32(seed),
  };
  vm.createContext(sandbox);
  vm.runInContext('Math.random = __random; delete this.__random;', sandbox);
  vm.runInContext(src, sandbox, { filename: 't-rex-runner/index.js' });
  const Runner = window.Runner;
  Runner.imageSprite = await loadImage(path.join(gameDir, 'assets', 'default_100_percent', '100-offline-sprite.png'));
  return { Runner, ...window.DinoGame, spriteDef: Runner.spriteDefinition.LDPI };
}

async function main() {
  const args = parseArgs();
  const lib = await loadGame(args.seed);
  const ctx = setupObsContext(createCanvas(lib.Runner.defaultDimensions.WIDTH, lib.Runner.defaultDimensions.HEIGHT).getContext('2d'));
  const router = makeRouter();
  const n = args.envs;
  const frameSize = OBS.height * OBS.width;
  const games = Array.from({ length: n }, () => new Game(lib, router));

  const headerSize = 6 * n;
  const reply = Buffer.alloc(headerSize + n * frameSize);

  function writeFrames() {
    for (let e = 0; e < n; e++) games[e].observe(ctx, reply, headerSize + e * frameSize);
  }

  function step(actions) {
    for (let e = 0; e < n; e++) {
      const g = games[e];
      const done = g.step(actions[e]);
      const truncated = !done && args.maxScore > 0 && g.score() >= args.maxScore;
      reply[e * 6] = done ? 1 : 0;
      reply[e * 6 + 1] = truncated ? 1 : 0;
      reply.writeFloatLE(g.score(), e * 6 + 2);
      if (done || truncated) g.reset();
    }
    writeFrames();
    process.stdout.write(reply);
  }

  writeFrames();
  process.stdout.write(reply.subarray(headerSize));

  // Команды могут прийти кусками; собираем их в буфер.
  let pending = Buffer.alloc(0);
  process.stdin.on('data', (chunk) => {
    pending = Buffer.concat([pending, chunk]);
    while (pending.length > 0) {
      const op = String.fromCharCode(pending[0]);
      if (op === 'Q') process.exit(0);
      if (op !== 'S') throw new Error('Неизвестная команда ' + op);
      if (pending.length < 1 + n) break;
      step(pending.subarray(1, 1 + n));
      pending = pending.subarray(1 + n);
    }
  });
  process.stdin.on('end', () => process.exit(0));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
