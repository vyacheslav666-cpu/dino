// NEAT — нейроэволюция нарастающих топологий (Stanley, Miikkulainen, 2002).
// Сети начинают с прямых связей «вход → выход» и сами обрастают нейронами.
// Похожие сети объединяются в виды, и соревнуются в основном внутри вида:
// так новая структура успевает дообучиться, прежде чем её вытеснят.
(function () {
  'use strict';

  const DEFAULTS = {
    popSize: 150,
    inputs: 1,
    outputs: 1,
    // Расстояние между сетями: c1·лишние + c2·несовпавшие + c3·разница весов.
    c1: 1,
    c2: 1,
    c3: 0.4,
    compatThreshold: 1,
    targetSpecies: 8,
    // Мутации.
    weightMutateRate: 0.8,
    weightPerturbRate: 0.9,
    weightPower: 0.5,
    weightMax: 8,
    addConnRate: 0.1,
    addNodeRate: 0.03,
    toggleRate: 0.01,
    // Размножение.
    crossoverRate: 0.75,
    interspeciesRate: 0.001,
    survivalRate: 0.2,
    eliteMinSize: 5,
    stagnation: 15,
  };

  const sigmoid = (x) => 1 / (1 + Math.exp(-4.9 * x));
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const randWeight = () => Math.random() * 4 - 2;

  function gauss() {
    let u = 0;
    while (!u) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  // Общий реестр инноваций: одна и та же новая связь (или нейрон на месте
  // одной и той же связи) получает один номер во всех геномах. По этим
  // номерам гены выравниваются при скрещивании и сравнении.
  class Innovations {
    constructor(firstNodeId) {
      this.conns = new Map();
      this.splits = new Map();
      this.nextInnov = 0;
      this.nextNode = firstNodeId;
    }

    conn(from, to) {
      const key = from + '>' + to;
      if (!this.conns.has(key)) this.conns.set(key, this.nextInnov++);
      return this.conns.get(key);
    }

    split(innov) {
      if (!this.splits.has(innov)) this.splits.set(innov, this.nextNode++);
      return this.splits.get(innov);
    }

    freshNode() {
      return this.nextNode++;
    }
  }

  // Узлы: 0..nIn-1 — входы (последний — смещение, всегда 1),
  // nIn..nIn+nOut-1 — выходы, дальше — скрытые нейроны.
  class Genome {
    constructor(pop, hidden, conns) {
      this.pop = pop;
      this.hidden = hidden;
      this.conns = conns; // отсортированы по innov
      this.fitness = 0;
      this.net = null;
    }

    clone() {
      return new Genome(this.pop, this.hidden.slice(), this.conns.map((c) => ({ ...c })));
    }

    toJSON() {
      return { hidden: this.hidden, conns: this.conns };
    }

    isInput(id) {
      return id < this.pop.nIn;
    }

    isOutput(id) {
      return id >= this.pop.nIn && id < this.pop.nIn + this.pop.nOut;
    }

    mutate() {
      const cfg = this.pop.cfg;
      if (Math.random() < cfg.weightMutateRate) this.mutateWeights();
      if (Math.random() < cfg.addConnRate) this.addConnection();
      if (Math.random() < cfg.addNodeRate) this.addNode();
      if (Math.random() < cfg.toggleRate && this.conns.length) {
        const c = pick(this.conns);
        c.enabled = !c.enabled;
      }
      this.net = null;
    }

    mutateWeights() {
      const cfg = this.pop.cfg;
      for (const c of this.conns) {
        if (Math.random() < cfg.weightPerturbRate) c.weight += gauss() * cfg.weightPower;
        else c.weight = randWeight();
        c.weight = Math.max(-cfg.weightMax, Math.min(cfg.weightMax, c.weight));
      }
    }

    // Есть ли путь from → … → to по связям (включённым или нет).
    reaches(from, to) {
      const seen = new Set();
      const stack = [from];
      while (stack.length) {
        const id = stack.pop();
        if (id === to) return true;
        if (seen.has(id)) continue;
        seen.add(id);
        for (const c of this.conns) if (c.from === id) stack.push(c.to);
      }
      return false;
    }

    // Сети только прямого распространения: связь, замыкающая цикл, запрещена.
    addConnection() {
      const { nIn, nOut } = this.pop;
      const sources = [...Array(nIn).keys(), ...this.hidden];
      const targets = [...this.hidden, ...Array.from({ length: nOut }, (_, i) => nIn + i)];
      for (let attempt = 0; attempt < 20; attempt++) {
        const from = pick(sources);
        const to = pick(targets);
        if (from === to) continue;
        const existing = this.conns.find((c) => c.from === from && c.to === to);
        if (existing) {
          if (!existing.enabled) {
            existing.enabled = true;
            return;
          }
          continue;
        }
        if (this.reaches(to, from)) continue;
        this.insertConn({
          innov: this.pop.innov.conn(from, to),
          from,
          to,
          weight: randWeight(),
          enabled: true,
        });
        return;
      }
    }

    // Новый нейрон встаёт посреди существующей связи.
    addNode() {
      const enabled = this.conns.filter((c) => c.enabled);
      if (!enabled.length) return;
      const c = pick(enabled);
      c.enabled = false;
      let id = this.pop.innov.split(c.innov);
      if (this.hidden.includes(id)) id = this.pop.innov.freshNode();
      this.hidden.push(id);
      const innov = this.pop.innov;
      this.insertConn({ innov: innov.conn(c.from, id), from: c.from, to: id, weight: 1, enabled: true });
      this.insertConn({ innov: innov.conn(id, c.to), from: id, to: c.to, weight: c.weight, enabled: true });
    }

    insertConn(conn) {
      let i = this.conns.length;
      while (i > 0 && this.conns[i - 1].innov > conn.innov) i--;
      this.conns.splice(i, 0, conn);
    }

    // Схема вычисления: нейроны в порядке зависимостей.
    build() {
      const { nIn, nOut } = this.pop;
      const outputs = Array.from({ length: nOut }, (_, i) => nIn + i);
      const incoming = new Map();
      for (const id of [...this.hidden, ...outputs]) incoming.set(id, []);
      for (const c of this.conns) if (c.enabled) incoming.get(c.to).push(c);
      const order = [];
      const done = new Set();
      const visit = (id) => {
        if (id < nIn || done.has(id)) return;
        done.add(id);
        for (const c of incoming.get(id)) visit(c.from);
        order.push(id);
      };
      for (const id of [...this.hidden, ...outputs]) visit(id);
      this.net = { order, incoming, outputs, values: new Map() };
      return this.net;
    }

    activate(inputs) {
      const net = this.net || this.build();
      const v = net.values;
      for (let i = 0; i < inputs.length; i++) v.set(i, inputs[i]);
      v.set(inputs.length, 1); // смещение
      for (const id of net.order) {
        let sum = 0;
        for (const c of net.incoming.get(id)) sum += (v.get(c.from) || 0) * c.weight;
        v.set(id, sigmoid(sum));
      }
      return net.outputs.map((id) => v.get(id));
    }
  }

  // a — более приспособленный родитель: структура берётся от него,
  // совпавшие гены — случайно от любого из двух.
  function crossover(a, b) {
    const byInnov = new Map(b.conns.map((c) => [c.innov, c]));
    const conns = a.conns.map((ca) => {
      const cb = byInnov.get(ca.innov);
      const child = { ...(cb && Math.random() < 0.5 ? cb : ca) };
      if (cb) child.enabled = ca.enabled && cb.enabled ? true : Math.random() > 0.75;
      return child;
    });
    return new Genome(a.pop, a.hidden.slice(), conns);
  }

  function distance(a, b, cfg) {
    let i = 0;
    let j = 0;
    let matching = 0;
    let weightDiff = 0;
    let disjoint = 0;
    while (i < a.conns.length && j < b.conns.length) {
      const ia = a.conns[i].innov;
      const ib = b.conns[j].innov;
      if (ia === ib) {
        matching++;
        weightDiff += Math.abs(a.conns[i].weight - b.conns[j].weight);
        i++;
        j++;
      } else if (ia < ib) {
        disjoint++;
        i++;
      } else {
        disjoint++;
        j++;
      }
    }
    const excess = a.conns.length - i + (b.conns.length - j);
    const longest = Math.max(a.conns.length, b.conns.length);
    const n = longest < 20 ? 1 : longest;
    return (cfg.c1 * excess + cfg.c2 * disjoint) / n + cfg.c3 * (matching ? weightDiff / matching : 0);
  }

  class Population {
    constructor(options) {
      this.cfg = { ...DEFAULTS, ...options };
      this.nIn = this.cfg.inputs + 1;
      this.nOut = this.cfg.outputs;
      this.innov = new Innovations(this.nIn + this.nOut);
      this.threshold = this.cfg.compatThreshold;
      this.generation = 1;
      this.species = [];
      this.nextSpeciesId = 1;
      this.best = null;
      this.history = [];
      this.genomes = Array.from({ length: this.cfg.popSize }, () => this.initialGenome());
      this.speciate();
    }

    // Все входы напрямую соединены со всеми выходами.
    initialGenome() {
      const conns = [];
      for (let o = 0; o < this.nOut; o++) {
        for (let i = 0; i < this.nIn; i++) {
          const to = this.nIn + o;
          conns.push({ innov: this.innov.conn(i, to), from: i, to, weight: randWeight(), enabled: true });
        }
      }
      conns.sort((x, y) => x.innov - y.innov);
      return new Genome(this, [], conns);
    }

    fromJSON(data) {
      return new Genome(this, data.hidden.slice(), data.conns.map((c) => ({ ...c })));
    }

    speciate() {
      for (const s of this.species) {
        s.rep = pick(s.members);
        s.members = [];
      }
      for (const g of this.genomes) {
        let home = this.species.find((s) => distance(g, s.rep, this.cfg) < this.threshold);
        if (!home) {
          home = { id: this.nextSpeciesId++, rep: g, members: [], best: -Infinity, stagnant: 0 };
          this.species.push(home);
        }
        home.members.push(g);
        g.species = home;
      }
      this.species = this.species.filter((s) => s.members.length);
      // Порог подстраивается, чтобы число видов держалось около целевого.
      const target = this.cfg.targetSpecies;
      if (this.species.length < target - 2) this.threshold = Math.max(0.2, this.threshold - 0.1);
      else if (this.species.length > target + 2) this.threshold += 0.1;
    }

    // Вызывается, когда у всех геномов выставлен fitness.
    evolve() {
      const cfg = this.cfg;
      const ranked = this.genomes.slice().sort((x, y) => y.fitness - x.fitness);
      const champion = ranked[0];
      const mean = ranked.reduce((sum, g) => sum + g.fitness, 0) / ranked.length;
      if (!this.best || champion.fitness >= this.best.fitness) {
        this.best = champion.clone();
        this.best.fitness = champion.fitness;
        this.best.generation = this.generation;
      }
      this.history.push({
        generation: this.generation,
        best: champion.fitness,
        mean,
        species: this.species.length,
        nodes: champion.hidden.length,
      });

      for (const s of this.species) {
        s.members.sort((x, y) => y.fitness - x.fitness);
        if (s.members[0].fitness > s.best) {
          s.best = s.members[0].fitness;
          s.stagnant = 0;
        } else {
          s.stagnant++;
        }
        // Разделённая приспособленность: средняя по виду.
        s.share = s.members.reduce((sum, g) => sum + g.fitness, 0) / s.members.length;
      }

      // Застоявшиеся виды вымирают, кроме двух лучших.
      const byBest = this.species.slice().sort((x, y) => y.best - x.best);
      const alive = byBest.filter((s, i) => i < 2 || s.stagnant < cfg.stagnation);

      const total = alive.reduce((sum, s) => sum + s.share, 0);
      const quotas = alive.map((s) => (total > 0 ? (s.share / total) * cfg.popSize : cfg.popSize / alive.length));
      const counts = quotas.map(Math.floor);
      let left = cfg.popSize - counts.reduce((a, b) => a + b, 0);
      alive
        .map((s, i) => i)
        .sort((x, y) => quotas[y] - counts[y] - (quotas[x] - counts[x]))
        .slice(0, left)
        .forEach((i) => counts[i]++);

      const everyone = alive.flatMap((s) => s.members);
      // Чемпион поколения переходит без изменений и идёт первым —
      // его динозавр рисуется ярче остальных.
      const next = [champion.clone()];
      alive.forEach((s, k) => {
        let n = counts[k];
        if (!n) return;
        if (s.members[0] === champion) {
          n--;
        } else if (s.members.length >= cfg.eliteMinSize) {
          next.push(s.members[0].clone());
          n--;
        }
        const parents = s.members.slice(0, Math.max(1, Math.ceil(s.members.length * cfg.survivalRate)));
        while (n-- > 0) {
          let child;
          if (Math.random() < cfg.crossoverRate) {
            const a = pick(parents);
            const b = Math.random() < cfg.interspeciesRate ? pick(everyone) : pick(parents);
            child = a.fitness >= b.fitness ? crossover(a, b) : crossover(b, a);
          } else {
            child = pick(parents).clone();
          }
          child.mutate();
          next.push(child);
        }
      });
      while (next.length < cfg.popSize) {
        const g = champion.clone();
        g.mutate();
        next.push(g);
      }

      this.species = alive;
      this.genomes = next.slice(0, cfg.popSize);
      this.generation++;
      this.speciate();
    }
  }

  window.NEAT = { Population, Genome, DEFAULTS };
})();
