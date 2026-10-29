// Одна партия игры для обучения с подкреплением. Общий код для среды в Node
// (dino_env.js) и для страницы в браузере (web/), чтобы управление и картинка,
// которую видит сеть, совпадали в обоих местах.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DinoRL = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Классы игры спрятаны внутри её модуля. Эти правки выставляют их наружу
  // и отключают автозапуск, сам файл игры не меняется.
  const PATCHES = [
    ["window['Runner'] = Runner;", "window['Runner'] = Runner; window.DinoGame = { Trex: Trex, Horizon: Horizon, checkForCollision: checkForCollision };"],
    ["document.addEventListener('DOMContentLoaded', onDocumentLoad);", ''],
  ];

  function patchSource(src) {
    for (const [from, to] of PATCHES) {
      if (!src.includes(from)) throw new Error('Не нашёл в коде игры: ' + from);
      src = src.replace(from, to);
    }
    return src;
  }

  const ACTION = { NONE: 0, JUMP: 1, DUCK: 2 };
  const STEP_MS = 1000 / 60;
  const SKIP = 2; // кадров игры на одно решение сети
  // Кадр для сети: сцена 600×150 уменьшается ровно в 4 раза, сверху
  // добавляются 8 пустых строк исходной картинки и 2 снизу — получается 150×40.
  const OBS = { height: 40, width: 150, stack: 4, scale: 4, padTop: 8 };

  // Игра рисует прямо во время обновления. Объекты игры получают этот
  // посредник вместо настоящего контекста: пока target пуст, рисование
  // выключено, а для снимка его направляют в нужный холст.
  function makeRouter() {
    const router = {
      target: null,
      drawImage() { if (router.target) router.target.drawImage(...arguments); },
      save() { if (router.target) router.target.save(); },
      restore() { if (router.target) router.target.restore(); },
      translate(x, y) { if (router.target) router.target.translate(x, y); },
      get globalAlpha() { return router.target ? router.target.globalAlpha : 1; },
      set globalAlpha(v) { if (router.target) router.target.globalAlpha = v; },
    };
    return router;
  }

  class Game {
    // lib: { Runner, Trex, Horizon, checkForCollision, spriteDef }
    constructor(lib, router) {
      this.lib = lib;
      this.router = router;
      const DIM = lib.Runner.defaultDimensions;
      this.canvas = { width: DIM.WIDTH, height: DIM.HEIGHT, getContext: () => router };
      this.reset();
    }

    reset() {
      const { Runner, Trex, Horizon, spriteDef } = this.lib;
      const RC = Runner.config;
      this.horizon = new Horizon(this.canvas, spriteDef, Runner.defaultDimensions, RC.GAP_COEFFICIENT);
      this.trex = new Trex(this.canvas, spriteDef.TREX);
      this.trex.xPos = Trex.config.START_X_POS;
      this.trex.update(0, Trex.status.RUNNING);
      this.speed = RC.SPEED;
      this.distance = 0;
      this.time = 0;
      this.prevAction = ACTION.NONE;
      this.crashed = false;
    }

    score() {
      return Math.round(Math.ceil(this.distance) * 0.025); // DistanceMeter.getActualDistance
    }

    // Действие — какая клавиша зажата. Повторяем Runner.onKeyDown/onKeyUp:
    // отпустили пробел в прыжке — прыжок короче; «вниз» в воздухе — быстрое
    // падение, на земле — пригнуться.
    press(action) {
      const t = this.trex;
      const prev = this.prevAction;
      if (prev === ACTION.JUMP && action !== ACTION.JUMP && t.jumping) t.endJump();
      if (prev === ACTION.DUCK && action !== ACTION.DUCK) {
        t.speedDrop = false;
        t.setDuck(false);
      }
      if (action === ACTION.JUMP && !t.jumping && !t.ducking) {
        t.startJump(this.speed);
      } else if (action === ACTION.DUCK) {
        if (t.jumping) {
          if (prev !== ACTION.DUCK) t.setSpeedDrop();
        } else if (!t.ducking) {
          t.setDuck(true);
        }
      }
      this.prevAction = action;
    }

    // Один кадр игры — тот же порядок, что в Runner.update. Возвращает true при столкновении.
    frame() {
      const { Runner, checkForCollision } = this.lib;
      const RC = Runner.config;
      this.time += STEP_MS;
      const hasObstacles = this.time > RC.CLEAR_TIME;
      if (this.trex.jumping) this.trex.updateJump(STEP_MS);
      this.horizon.update(STEP_MS, this.speed, hasObstacles, false);
      // Игра проверяет столкновение только с первым препятствием в списке.
      const first = this.horizon.obstacles[0];
      if (hasObstacles && first && checkForCollision(first, this.trex)) {
        this.crashed = true;
        return true;
      }
      this.distance += this.speed;
      if (this.speed < RC.MAX_SPEED) this.speed += RC.ACCELERATION;
      this.trex.update(STEP_MS);
      return false;
    }

    // Одно решение сети: действие и SKIP кадров. Возвращает true при столкновении.
    step(action) {
      this.press(action);
      for (let k = 0; k < SKIP; k++) if (this.frame()) return true;
      return false;
    }

    // Рисует текущую сцену в ctx. Масштаб задаёт вызывающий.
    draw(ctx) {
      this.router.target = ctx;
      this.horizon.update(0, this.speed, this.time > this.lib.Runner.config.CLEAR_TIME, false);
      this.trex.update(0);
      this.router.target = null;
    }

    // Кадр для сети в out[offset…]: OBS.height × OBS.width байт, оттенки
    // серого, фон 0. ctx — холст в полный размер игры (см. setupObsContext).
    // Сцена рисуется 1:1 без сглаживания, а уменьшается здесь усреднением
    // блоков: встроенное сглаживание холста в Node и в браузере даёт разную
    // картинку, а так она совпадает до байта.
    observe(ctx, out, offset) {
      const DIM = this.lib.Runner.defaultDimensions;
      const W = DIM.WIDTH;
      const H = DIM.HEIGHT;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, W, H);
      this.draw(ctx);
      const rgba = ctx.getImageData(0, 0, W, H).data;
      const s = OBS.scale;
      for (let r = 0; r < OBS.height; r++) {
        const y0 = r * s - OBS.padTop;
        for (let c = 0; c < OBS.width; c++) {
          let sum = 0;
          for (let y = Math.max(y0, 0); y < Math.min(y0 + s, H); y++) {
            let i = (y * W + c * s) * 4;
            for (let x = 0; x < s; x++, i += 4) sum += 255 - rgba[i];
          }
          out[offset + r * OBS.width + c] = (sum + (s * s) / 2) / (s * s) | 0;
        }
      }
    }
  }

  // Холст для снимков: размер игры, без сглаживания.
  function setupObsContext(ctx) {
    ctx.imageSmoothingEnabled = false;
    return ctx;
  }

  return { PATCHES, patchSource, ACTION, STEP_MS, SKIP, OBS, makeRouter, Game, setupObsContext };
});
