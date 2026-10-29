// Схема «мозга»: слои Q-сети слева направо и что в них происходит прямо
// сейчас. Свёрточные слои — карты признаков (ярче — сильнее сработал фильтр),
// полносвязные — сетки по 256 нейронов, линии — самые сильные вклады
// нейронов в выходы, как связи на схеме NEAT.
window.DinoBrain = (function () {
  'use strict';

  const W = 1000;
  const H = 330;
  const TOP = 12;
  const AREA = 250; // высота области со слоями, ниже подписи
  const GAP = 42;   // между колонками
  const TOP_LINKS = 10; // сколько связей рисовать к каждому выходу

  // Тёмный → синий → белый, как светлое на тёмном во входных кадрах.
  const LUT = (function () {
    const stops = [[0, [24, 24, 28]], [0.6, [47, 111, 219]], [1, [240, 244, 255]]];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      const k = t <= stops[1][0] ? 0 : 1;
      const [t0, c0] = stops[k];
      const [t1, c1] = stops[k + 1];
      const u = (t - t0) / (t1 - t0);
      for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * u;
    }
    return lut;
  })();

  // Слои с картами. Вход хранится по кадрам [k][y][x], выходы свёрток в
  // TensorFlow.js — [y][x][канал].
  const MAP_LAYERS = [
    { key: 'input', n: 4, h: 40, w: 150, scale: 0.6, cols: 1, gap: 6, name: 'Вход', dims: '4 кадра 150×40' },
    { key: 'c1', n: 32, h: 12, w: 49, scale: 1, cols: 2, gap: 3, name: 'Свёртка 1', dims: '32 карты 12×49' },
    { key: 'c2', n: 64, h: 5, w: 23, scale: 2, cols: 4, gap: 3, name: 'Свёртка 2', dims: '64 карты 5×23' },
    { key: 'c3', n: 64, h: 3, w: 21, scale: 2, cols: 4, gap: 3, name: 'Свёртка 3', dims: '64 карты 3×21' },
  ];
  const DENSE = { side: 16, cell: 5, pitch: 6, caption: 17, between: 16 };
  const OUT_W = 120;

  function layout() {
    let x = 10;
    for (const L of MAP_LAYERS) {
      const rows = Math.ceil(L.n / L.cols);
      L.cw = L.w * L.scale;
      L.ch = L.h * L.scale;
      L.bw = L.cols * L.cw + (L.cols - 1) * L.gap;
      L.bh = rows * L.ch + (rows - 1) * L.gap;
      L.x = x;
      L.y = TOP + (AREA - L.bh) / 2;
      x += L.bw + GAP;
    }
    const gridPx = DENSE.side * DENSE.pitch - 1;
    const total = 2 * (DENSE.caption + gridPx) + DENSE.between;
    const y0 = TOP + (AREA - total) / 2;
    const dense = {
      x, w: gridPx,
      v: { y: y0 + DENSE.caption, caption: 'ценность V' },
      a: { y: y0 + 2 * DENSE.caption + gridPx + DENSE.between, caption: 'преимущество A' },
    };
    x += gridPx + GAP;
    const vCy = dense.v.y + gridPx / 2;
    const aCy = dense.a.y + gridPx / 2;
    const out = { x: x + 8, v: vCy, a: [aCy - 32, aCy, aCy + 32] };
    return { dense, out, gridPx };
  }

  function create(canvas, heads) {
    const { dense, out, gridPx } = layout();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext('2d');

    // Карты слоя рисуются в родном размере на служебный холст (все карты
    // столбиком), потом растягиваются в свои клетки без сглаживания.
    for (const L of MAP_LAYERS) {
      L.atlas = document.createElement('canvas');
      L.atlas.width = L.w;
      L.atlas.height = L.h * L.n;
      L.actx = L.atlas.getContext('2d');
      L.img = new ImageData(L.w, L.h * L.n);
    }

    function colors() {
      const s = getComputedStyle(document.documentElement);
      const v = (name) => s.getPropertyValue(name).trim();
      return { panel: v('--panel'), ink: v('--ink'), muted: v('--muted'), line: v('--line'), pos: v('--accent'), neg: '#e0703a' };
    }

    function fillAtlas(L, data) {
      const px = L.img.data;
      const plane = L.h * L.w;
      if (L.key === 'input') {
        for (let i = 0; i < L.n * plane; i++) {
          px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = data[i];
          px[i * 4 + 3] = 255;
        }
      } else {
        let max = 1e-6;
        for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
        for (let c = 0; c < L.n; c++) {
          for (let p = 0; p < plane; p++) {
            const t = Math.min(255, (data[p * L.n + c] / max) * 255) | 0;
            const o = (c * plane + p) * 4;
            px[o] = LUT[t * 3];
            px[o + 1] = LUT[t * 3 + 1];
            px[o + 2] = LUT[t * 3 + 2];
            px[o + 3] = 255;
          }
        }
      }
      L.actx.putImageData(L.img, 0, 0);
    }

    function drawMaps(L, has) {
      for (let k = 0; k < L.n; k++) {
        const col = k % L.cols;
        const row = (k / L.cols) | 0;
        const x = L.x + col * (L.cw + L.gap);
        const y = L.y + row * (L.ch + L.gap);
        if (has) ctx.drawImage(L.atlas, 0, k * L.h, L.w, L.h, x, y, L.cw, L.ch);
        else {
          ctx.fillStyle = 'rgb(24,24,28)';
          ctx.fillRect(x, y, L.cw, L.ch);
        }
      }
    }

    function drawGrid(g, data) {
      let max = 1e-6;
      if (data) for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
      for (let j = 0; j < DENSE.side * DENSE.side; j++) {
        const t = data ? Math.min(255, (data[j] / max) * 255) | 0 : 0;
        ctx.fillStyle = `rgb(${LUT[t * 3]},${LUT[t * 3 + 1]},${LUT[t * 3 + 2]})`;
        ctx.fillRect(dense.x + (j % DENSE.side) * DENSE.pitch, g.y + ((j / DENSE.side) | 0) * DENSE.pitch, DENSE.cell, DENSE.cell);
      }
    }

    function neuronCenter(g, j) {
      return [
        dense.x + (j % DENSE.side) * DENSE.pitch + DENSE.cell / 2,
        g.y + ((j / DENSE.side) | 0) * DENSE.pitch + DENSE.cell / 2,
      ];
    }

    // Вклад нейрона j в выход o: активация × вес. Рисуем самые большие по модулю.
    function links(hidden, weights, nOut) {
      const res = [];
      for (let o = 0; o < nOut; o++) {
        const c = [];
        for (let j = 0; j < hidden.length; j++) {
          const v = hidden[j] * weights[j * nOut + o];
          if (v !== 0) c.push([j, o, v]);
        }
        c.sort((p, q) => Math.abs(q[2]) - Math.abs(p[2]));
        res.push(...c.slice(0, TOP_LINKS));
      }
      return res;
    }

    function bundle(x0, y0a, y0b, x1, y1a, y1b, color) {
      ctx.beginPath();
      ctx.moveTo(x0, y0a);
      ctx.bezierCurveTo((x0 + x1) / 2, y0a, (x0 + x1) / 2, y1a, x1, y1a);
      ctx.lineTo(x1, y1b);
      ctx.bezierCurveTo((x0 + x1) / 2, y1b, (x0 + x1) / 2, y0b, x0, y0b);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    }

    function node(x, y, r, fill, stroke) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }

    function draw(acts, q, action, names) {
      const col = colors();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = col.panel;
      ctx.fillRect(0, 0, W, H);
      const has = !!(acts && acts.c3);

      // Потоки между слоями.
      ctx.globalAlpha = 0.5;
      for (let i = 0; i + 1 < MAP_LAYERS.length; i++) {
        const a = MAP_LAYERS[i];
        const b = MAP_LAYERS[i + 1];
        bundle(a.x + a.bw + 3, a.y, a.y + a.bh, b.x - 3, b.y, b.y + b.bh, col.line);
      }
      const c3 = MAP_LAYERS[MAP_LAYERS.length - 1];
      for (const g of [dense.v, dense.a]) {
        bundle(c3.x + c3.bw + 3, c3.y, c3.y + c3.bh, dense.x - 3, g.y, g.y + gridPx, col.line);
      }
      ctx.globalAlpha = 1;

      for (const L of MAP_LAYERS) {
        if (has) fillAtlas(L, acts[L.key]);
        drawMaps(L, has);
      }

      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillStyle = col.muted;
      for (const g of [dense.v, dense.a]) ctx.fillText(g.caption, dense.x, g.y - 4);
      drawGrid(dense.v, has ? acts.hv : null);
      drawGrid(dense.a, has ? acts.ha : null);

      // Связи к выходам.
      if (has) {
        const all = [
          ...links(acts.hv, heads.value, 1).map(([j, , v]) => [neuronCenter(dense.v, j), [out.x, out.v], v]),
          ...links(acts.ha, heads.advantage, 3).map(([j, o, v]) => [neuronCenter(dense.a, j), [out.x, out.a[o]], v]),
        ];
        const maxAbs = Math.max(1e-6, ...all.map((l) => Math.abs(l[2])));
        ctx.globalAlpha = 0.7;
        for (const [[x0, y0], [x1, y1], v] of all) {
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1 - 9, y1);
          ctx.lineWidth = 0.5 + 3 * Math.abs(v) / maxAbs;
          ctx.strokeStyle = v > 0 ? col.pos : col.neg;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // V прибавляется к оценке каждого действия.
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = col.muted;
      for (const y of out.a) {
        ctx.beginPath();
        ctx.moveTo(out.x, out.v + 9);
        ctx.lineTo(out.x, y - 9);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      node(out.x, out.v, 8, col.panel, col.muted);
      ctx.fillStyle = col.ink;
      ctx.font = '15px system-ui, sans-serif';
      ctx.fillText('V ' + (has ? acts.v.toFixed(2) : '—'), out.x + 14, out.v + 4);
      names.forEach((name, i) => {
        const chosen = has && i === action;
        node(out.x, out.a[i], 8, chosen ? col.pos : col.panel, chosen ? col.pos : col.muted);
        ctx.fillStyle = chosen ? col.ink : col.muted;
        ctx.font = (chosen ? '600 ' : '') + '15px system-ui, sans-serif';
        ctx.fillText(name + ' ' + (has ? q[i].toFixed(2) : ''), out.x + 14, out.a[i] + 4);
      });

      // Подписи слоёв.
      const labels = [
        ...MAP_LAYERS.map((L) => [L.x + L.bw / 2, L.name, L.dims]),
        [dense.x + gridPx / 2, 'Полносвязный', '2 × 256 нейронов'],
        [out.x + OUT_W / 2 - 14, 'Выход', 'Q = V + A − ср. A'],
      ];
      ctx.textAlign = 'center';
      for (const [x, name, dims] of labels) {
        ctx.fillStyle = col.ink;
        ctx.font = '600 15px system-ui, sans-serif';
        ctx.fillText(name, x, TOP + AREA + 28);
        ctx.fillStyle = col.muted;
        ctx.font = '13px system-ui, sans-serif';
        ctx.fillText(dims, x, TOP + AREA + 47);
      }
    }

    return { draw };
  }

  return { create };
})();
