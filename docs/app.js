/* FFT Accelerated: charts and derived numbers. All values come from window.FFT_DATA (data.js),
   which is generated from the repo's results.csv files. */
(function () {
  "use strict";
  var D = window.FFT_DATA;
  if (!D) return;

  var NS = "http://www.w3.org/2000/svg";
  var css = getComputedStyle(document.documentElement);
  function v(name) { return css.getPropertyValue(name).trim(); }
  var COL = {
    omp: v("--series-1"), mpi: v("--series-2"), cuda: v("--series-3"),
    jaxcpu: v("--series-4"), jaxgpu: v("--series-5"), serial: v("--color-text")
  };

  // ---------- data helpers ----------
  var serialMap = toMap(D.serial);
  var NS_LIST = D.serial.map(function (r) { return r[0]; });
  function toMap(arr) { var m = {}; arr.forEach(function (r) { m[r[0]] = r[1]; }); return m; }
  function groupMaps(g) { var o = {}; Object.keys(g).forEach(function (k) { o[k] = toMap(g[k]); }); return o; }
  var G = { openmp: groupMaps(D.openmp), mpi: groupMaps(D.mpi), cuda: groupMaps(D.cuda) };
  var JAX = { gpu: toMap(D.jax.gpu), cpu: toMap(D.jax.cpu) };
  function sortedKeys(o) { return Object.keys(o).sort(function (a, b) { return +a - +b; }); }

  function bestAt(groups, n) {
    var best = null;
    sortedKeys(groups).forEach(function (k) {
      var t = groups[k][n];
      if (t != null && (!best || t < best.t)) best = { k: k, t: t };
    });
    return best;
  }
  function bestSeries(groups) { return NS_LIST.map(function (n) { var b = bestAt(groups, n); return { n: n, t: b.t, k: b.k }; }); }
  function plainSeries(m) { return NS_LIST.map(function (n) { return { n: n, t: m[n] }; }); }

  var IMPL = [
    { id: "serial", name: "Serial", color: COL.serial, dashed: true, pts: plainSeries(serialMap), cfg: null },
    { id: "omp", name: "OpenMP", color: COL.omp, pts: bestSeries(G.openmp), cfg: "threads" },
    { id: "mpi", name: "MPI", color: COL.mpi, pts: bestSeries(G.mpi), cfg: "ranks" },
    { id: "cuda", name: "CUDA", color: COL.cuda, pts: bestSeries(G.cuda), cfg: "block size" },
    { id: "jaxcpu", name: "JAX (CPU)", color: COL.jaxcpu, pts: plainSeries(JAX.cpu), cfg: null },
    { id: "jaxgpu", name: "JAX (GPU)", color: COL.jaxgpu, pts: plainSeries(JAX.gpu), cfg: null }
  ];
  IMPL.forEach(function (s) { s.pts.forEach(function (p) { p.s = serialMap[p.n] / p.t; }); });

  function pow2(n) { return Math.round(Math.log2(n)); }
  function fmtX(x) { return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2); }
  function fmtMs(t) {
    if (t >= 1000) return t.toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (t >= 10) return t.toFixed(1);
    if (t >= 1) return t.toFixed(2);
    return t.toFixed(3);
  }
  function supN(n) { return "2<sup>" + pow2(n) + "</sup>"; }

  // ---------- derived stats ----------
  function peak(impl) {
    var b = null;
    impl.pts.forEach(function (p) { if (!b || p.s > b.s) b = p; });
    return b;
  }
  var byId = {}; IMPL.forEach(function (s) { byId[s.id] = s; });
  var N26 = NS_LIST[NS_LIST.length - 1];
  var stats = {
    "omp-peak": fmtX(peak(byId.omp).s),
    "mpi-peak": fmtX(peak(byId.mpi).s),
    "cuda-peak": fmtX(peak(byId.cuda).s),
    "jax-peak": fmtX(peak(byId.jaxgpu).s),
    "serial-26": fmtMs(serialMap[N26]),
    "omp-strong": (G.openmp["1"][N26] / G.openmp["32"][N26]).toFixed(2),
    "mpi-strong": (G.mpi["1"][N26] / G.mpi["32"][N26]).toFixed(2),
    "jax-vs-cuda": (JAX.gpu[N26] / bestAt(G.cuda, N26).t).toFixed(1),
    "jaxcpu-slow": (JAX.cpu[N26] / serialMap[N26]).toFixed(1)
  };
  document.querySelectorAll("[data-stat]").forEach(function (el) {
    var val = stats[el.getAttribute("data-stat")];
    if (val == null) return;
    var small = el.querySelector("small");
    if (small) { el.firstChild.nodeValue = val; } else { el.textContent = val; }
  });

  // ---------- svg helpers ----------
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function tooltip(container) {
    var t = document.createElement("div");
    t.className = "tooltip";
    t.setAttribute("role", "status");
    container.appendChild(t);
    return {
      show: function (html, x, y) {
        t.innerHTML = html;
        t.classList.add("on");
        var cw = container.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
        var left = x + 14; if (left + tw > cw) left = x - tw - 14; if (left < 0) left = Math.max(0, cw - tw);
        var top = y - th - 10; if (top < 0) top = y + 14;
        t.style.left = left + "px"; t.style.top = top + "px";
      },
      hide: function () { t.classList.remove("on"); }
    };
  }
  function row(color, label, value) {
    return '<div class="tt-row"><span class="sw" style="background:' + color + '"></span>' + label + "<b>" + value + "</b></div>";
  }
  function niceLogTicks(min, max) {
    var out = [], e0 = Math.floor(Math.log10(min)), e1 = Math.ceil(Math.log10(max));
    for (var e = e0; e <= e1; e++) out.push(Math.pow(10, e));
    return out;
  }
  function niceLinTicks(min, max, count) {
    var span = max - min, step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = span / count / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = []; for (var t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }
  function fmtLogTick(t) {
    if (t >= 1000) return (t / 1000) + "s";
    if (t >= 1) return t + "ms";
    return t + "ms";
  }

  // ---------- generic line chart ----------
  function lineChart(opts) {
    var host = document.getElementById(opts.id);
    var legend = document.getElementById(opts.legend);
    var hidden = {};
    var tip = tooltip(host);

    if (legend) {
      opts.series.forEach(function (s) {
        var b = document.createElement("button");
        b.type = "button";
        b.setAttribute("aria-pressed", "true");
        b.innerHTML = '<span class="lsw' + (s.dashed ? " dashed" : "") + '" style="background:' + s.color + '"></span>' + s.name;
        b.addEventListener("click", function () {
          hidden[s.id] = !hidden[s.id];
          b.setAttribute("aria-pressed", hidden[s.id] ? "false" : "true");
          draw();
        });
        legend.appendChild(b);
      });
    }

    function draw() {
      host.querySelectorAll("svg").forEach(function (n) { n.remove(); });
      var W = Math.max(300, host.clientWidth);
      var narrow = W < 520;
      var H = opts.height ? opts.height(W) : Math.round(Math.min(420, Math.max(280, W * 0.5)));
      var m = { t: 14, r: narrow ? 12 : 22, b: 44, l: narrow ? 46 : 58 };
      var iw = W - m.l - m.r, ih = H - m.t - m.b;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": opts.aria });
      host.insertBefore(svg, host.firstChild);

      var xs = opts.xs;
      var xLog = opts.xLog !== false;
      var xmin = xs[0], xmax = xs[xs.length - 1];
      function X(x) {
        return xLog ? m.l + (Math.log2(x) - Math.log2(xmin)) / (Math.log2(xmax) - Math.log2(xmin)) * iw
                    : m.l + (x - xmin) / (xmax - xmin) * iw;
      }
      var vis = opts.series.filter(function (s) { return !hidden[s.id]; });
      var allY = []; vis.forEach(function (s) { s.pts.forEach(function (p) { allY.push(p.y); }); });
      if (opts.extraY) allY = allY.concat(opts.extraY);
      if (!allY.length) allY = [1];
      var ymin, ymax, yticks, Y;
      if (opts.yLog) {
        ymin = Math.min.apply(null, allY) * 0.8; ymax = Math.max.apply(null, allY) * 1.25;
        yticks = niceLogTicks(ymin, ymax).filter(function (t) { return t >= ymin && t <= ymax; });
        Y = function (y) { return m.t + ih - (Math.log10(y) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin)) * ih; };
      } else {
        ymin = 0; ymax = Math.max.apply(null, allY) * 1.08;
        yticks = niceLinTicks(0, ymax, narrow ? 4 : 6);
        ymax = Math.max(ymax, yticks[yticks.length - 1]);
        Y = function (y) { return m.t + ih - (y - ymin) / (ymax - ymin) * ih; };
      }

      var grid = el("g", { class: "grid" }, svg);
      var axis = el("g", { class: "axis" }, svg);
      yticks.forEach(function (t) {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, grid);
        var tx = el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, axis);
        tx.textContent = opts.yFmt(t);
      });
      var step = narrow ? (opts.xStepNarrow || 4) : (opts.xStep || 2);
      xs.forEach(function (x, i) {
        if (i % step !== 0 && i !== xs.length - 1) return;
        var tx = el("text", { x: X(x), y: H - m.b + 18, "text-anchor": "middle" }, axis);
        tx.textContent = opts.xFmt(x);
      });
      el("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
      var xt = el("text", { class: "axis-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, svg);
      xt.textContent = opts.xTitle;
      var yt = el("text", { class: "axis-title", x: 0, y: 0, transform: "translate(11," + (m.t + ih / 2) + ") rotate(-90)", "text-anchor": "middle" }, svg);
      yt.textContent = opts.yTitle;

      if (opts.refY != null && opts.refY <= ymax) {
        el("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(opts.refY), y2: Y(opts.refY) }, svg);
        if (opts.refLabel) {
          var rl = el("text", { class: "ticklabel", x: W - m.r, y: Y(opts.refY) - 6, "text-anchor": "end" }, svg);
          rl.textContent = opts.refLabel;
        }
      }

      var groups = {};
      vis.forEach(function (s) {
        var g = el("g", { class: "series" }, svg);
        var d = s.pts.map(function (p, i) { return (i ? "L" : "M") + X(p.x).toFixed(1) + "," + Y(p.y).toFixed(1); }).join("");
        var path = el("path", { d: d, stroke: s.color }, g);
        if (s.dashed) path.setAttribute("stroke-dasharray", "5 4");
        groups[s.id] = g;
      });

      var cross = el("line", { class: "crosshair", y1: m.t, y2: m.t + ih, x1: -10, x2: -10, visibility: "hidden" }, svg);
      var dots = el("g", { class: "series" }, svg);
      var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      hit.style.cursor = "crosshair";

      function nearest(px) {
        var best = 0, bd = Infinity;
        xs.forEach(function (x, i) { var d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = i; } });
        return xs[best];
      }
      function move(ev) {
        var r = svg.getBoundingClientRect();
        var scale = W / r.width;
        var px = (ev.clientX - r.left) * scale;
        var x = nearest(px);
        cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("visibility", "visible");
        while (dots.firstChild) dots.removeChild(dots.firstChild);
        var rows = vis.map(function (s) {
          var p = s.pts.find(function (q) { return q.x === x; });
          if (!p) return null;
          el("circle", { cx: X(x), cy: Y(p.y), r: 4.5, fill: s.color }, dots);
          return { s: s, p: p };
        }).filter(Boolean).sort(function (a, b) { return b.p.y - a.p.y; });
        var html = '<div class="tt-head">' + opts.ttHead(x) + "</div>" +
          rows.map(function (o) { return row(o.s.color, o.s.name + (o.p.note ? ' <span class="muted">' + o.p.note + "</span>" : ""), opts.ttFmt(o.p.y)); }).join("");
        tip.show(html, X(x) / scale, (ev.clientY - r.top));
      }
      function leave() {
        cross.setAttribute("visibility", "hidden");
        while (dots.firstChild) dots.removeChild(dots.firstChild);
        tip.hide();
      }
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerdown", move);
      hit.addEventListener("pointerleave", leave);
    }
    draw();
    return draw;
  }

  // ---------- Figure 1: speedup ----------
  function cfgNote(s, p) { return s.cfg && p.k ? "(" + p.k + " " + s.cfg + ")" : ""; }
  var speedSeries = IMPL.map(function (s) {
    return { id: s.id, name: s.name, color: s.color, dashed: s.dashed,
      pts: s.pts.map(function (p) { return { x: p.n, y: p.s, note: cfgNote(s, p) }; }) };
  });
  var drawSpeed = lineChart({
    id: "chart-speedup", legend: "legend-speedup", series: speedSeries, xs: NS_LIST,
    aria: "Line chart of speedup over serial versus problem size for each implementation",
    xTitle: "Problem size n", yTitle: "Speedup ×",
    xFmt: function (x) { return "2^" + pow2(x); },
    yFmt: function (y) { return y + "×"; },
    ttHead: function (x) { return "n = 2^" + pow2(x) + " · " + x.toLocaleString("en-US"); },
    ttFmt: function (y) { return fmtX(y) + "×"; },
    refY: 1, refLabel: "serial = 1×"
  });

  // ---------- Figure 2: runtime ----------
  var runSeries = IMPL.map(function (s) {
    return { id: s.id, name: s.name, color: s.color, dashed: s.dashed,
      pts: s.pts.map(function (p) { return { x: p.n, y: p.t, note: cfgNote(s, p) }; }) };
  });
  var drawRun = lineChart({
    id: "chart-runtime", legend: "legend-runtime", series: runSeries, xs: NS_LIST, yLog: true,
    aria: "Log-log line chart of runtime in milliseconds versus problem size",
    xTitle: "Problem size n", yTitle: "Time per FFT",
    xFmt: function (x) { return "2^" + pow2(x); },
    yFmt: fmtLogTick,
    ttHead: function (x) { return "n = 2^" + pow2(x); },
    ttFmt: function (y) { return fmtMs(y) + " ms"; }
  });

  // ---------- Figure 4: weak scaling ----------
  var BASE = Math.pow(2, 21);
  function weak(groups) {
    var t1 = groups["1"][BASE];
    return sortedKeys(groups).map(function (p) {
      var n = BASE * +p, t = groups[p][n];
      return t == null ? null : { x: +p, y: 100 * t1 / t, note: "(n = 2^" + pow2(n) + ")" };
    }).filter(Boolean);
  }
  var weakSeries = [
    { id: "omp", name: "OpenMP threads", color: COL.omp, pts: weak(G.openmp) },
    { id: "mpi", name: "MPI ranks", color: COL.mpi, pts: weak(G.mpi) }
  ];
  var drawWeak = lineChart({
    id: "chart-weak", legend: "legend-weak", series: weakSeries, xs: weakSeries[0].pts.map(function (p) { return p.x; }),
    aria: "Line chart of weak scaling efficiency for OpenMP and MPI from 1 to 32 workers",
    xTitle: "Threads / ranks (p)", yTitle: "Efficiency %", xStep: 1, xStepNarrow: 1,
    xFmt: function (x) { return String(x); },
    yFmt: function (y) { return y + "%"; },
    ttHead: function (x) { return "p = " + x; },
    ttFmt: function (y) { return y.toFixed(1) + "%"; },
    refY: 100, refLabel: "ideal", extraY: [100],
    height: function (W) { return Math.round(Math.min(340, Math.max(240, W * 0.4))); }
  });

  // ---------- horizontal / vertical bar helper ----------
  function barChart(host, items, o) {
    host.querySelectorAll("svg").forEach(function (n) { n.remove(); });
    var tip = host._tip || (host._tip = tooltip(host));
    var W = Math.max(280, host.clientWidth);
    var narrow = W < 520;
    var H = o.height(W);
    var m = { t: 18, r: 12, b: 44, l: narrow ? 46 : 58 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": o.aria });
    host.insertBefore(svg, host.firstChild);
    var ymax = Math.max(o.minMax || 0, Math.max.apply(null, items.map(function (d) { return d.y; })) * 1.12);
    var ticks = niceLinTicks(0, ymax, narrow ? 4 : 5);
    ymax = Math.max(ymax, ticks[ticks.length - 1]);
    function Y(y) { return m.t + ih - y / ymax * ih; }
    var grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
    ticks.forEach(function (t) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, grid);
      el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, axis).textContent = o.yFmt(t);
    });
    if (o.refY != null && o.refY <= ymax) {
      el("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(o.refY), y2: Y(o.refY) }, svg);
      el("text", { class: "ticklabel", x: W - m.r, y: Y(o.refY) - 6, "text-anchor": "end" }, svg).textContent = o.refLabel;
    }
    var bw = iw / items.length, gap = Math.max(2, Math.min(18, bw * 0.28));
    items.forEach(function (d, i) {
      var x = m.l + i * bw + gap / 2, w = bw - gap, y = Y(d.y), h = m.t + ih - y;
      var r = Math.min(4, w / 2, h);
      var path = "M" + x + "," + (m.t + ih) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y +
        "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (m.t + ih) + "Z";
      var bar = el("path", { d: path, fill: o.color, class: "bar" }, svg);
      el("text", { class: "barlabel", x: x + w / 2, y: y - 6, "text-anchor": "middle" }, svg).textContent = o.label(d);
      el("text", { x: x + w / 2, y: H - m.b + 18, "text-anchor": "middle" }, axis).textContent = d.name;
      var hitR = el("rect", { x: m.l + i * bw, y: m.t, width: bw, height: ih, fill: "transparent" }, svg);
      function show(ev) {
        bar.setAttribute("opacity", "0.78");
        var r2 = svg.getBoundingClientRect(), s = W / r2.width;
        tip.show(o.tt(d), (x + w / 2) / s, y / s);
      }
      hitR.addEventListener("pointermove", show);
      hitR.addEventListener("pointerdown", show);
      hitR.addEventListener("pointerleave", function () { bar.removeAttribute("opacity"); tip.hide(); });
    });
    el("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
    el("text", { class: "axis-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = o.xTitle;
    el("text", { class: "axis-title", x: 0, y: 0, transform: "translate(11," + (m.t + ih / 2) + ") rotate(-90)", "text-anchor": "middle" }, svg).textContent = o.yTitle;
  }

  // ---------- Figure 3: explorer ----------
  var EXP = {
    openmp: { name: "OpenMP", color: COL.omp, unit: "threads", xTitle: "Threads" },
    mpi: { name: "MPI", color: COL.mpi, unit: "ranks", xTitle: "MPI processes" },
    cuda: { name: "CUDA", color: COL.cuda, unit: "threads / block", xTitle: "Block size" }
  };
  var expState = { k: "openmp", n: Math.pow(2, 21) };
  var sel = document.getElementById("explorer-n");
  NS_LIST.forEach(function (n) {
    var o = document.createElement("option");
    o.value = n; o.textContent = "2^" + pow2(n);
    if (n === expState.n) o.selected = true;
    sel.appendChild(o);
  });
  sel.addEventListener("change", function () { expState.n = +sel.value; drawExp(); });
  document.querySelectorAll("#explorer-backend button").forEach(function (b) {
    b.addEventListener("click", function () {
      expState.k = b.getAttribute("data-k");
      document.querySelectorAll("#explorer-backend button").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      drawExp();
    });
  });
  var expHost = document.getElementById("chart-explorer");
  function drawExp() {
    var meta = EXP[expState.k], groups = G[expState.k], n = expState.n, base = serialMap[n];
    var items = sortedKeys(groups).map(function (k) { var t = groups[k][n]; return { name: k, t: t, y: base / t }; });
    document.getElementById("explorer-sub").innerHTML =
      meta.name + " at n = " + supN(n) + ". Serial takes " + fmtMs(base) + " ms. The dashed line marks 1× (serial speed).";
    barChart(expHost, items, {
      color: meta.color, aria: meta.name + " speedup over serial by configuration at n = 2^" + pow2(n),
      xTitle: meta.xTitle, yTitle: "Speedup ×", refY: 1, refLabel: "1×", minMax: 1.2,
      yFmt: function (y) { return y + "×"; },
      label: function (d) { return fmtX(d.y) + "×"; },
      tt: function (d) {
        return '<div class="tt-head">' + meta.name + " · " + d.name + " " + meta.unit + "</div>" +
          row(meta.color, "Speedup", fmtX(d.y) + "×") + row("transparent", "Time", fmtMs(d.t) + " ms");
      },
      height: function (W) { return Math.round(Math.min(340, Math.max(240, W * 0.42))); }
    });
  }
  drawExp();

  // ---------- VTune bars (horizontal, static data from openmp/vtune_hotspots_summary.txt) ----------
  var VT = [
    { name: "Butterfly compute", fn: "fft._omp_fn.0", s: 180.740, p: 31.9, c: COL.omp },
    { name: "Team barrier wait", fn: "gomp_team_barrier_wait_end", s: 171.802, p: 30.3, c: v("--color-text-faint") },
    { name: "Simple barrier wait", fn: "gomp_simple_barrier_wait", s: 136.730, p: 24.1, c: v("--color-text-faint") },
    { name: "Bit reversal", fn: "bit_reverse._omp_fn.0", s: 42.546, p: 7.5, c: COL.omp },
    { name: "memcpy", fn: "__memcpy_evex_unaligned_erms", s: 14.230, p: 2.5, c: v("--color-text-faint") }
  ];
  var vtHost = document.getElementById("chart-vtune");
  var vtTip = tooltip(vtHost);
  function drawVT() {
    vtHost.querySelectorAll("svg").forEach(function (n) { n.remove(); });
    var W = Math.max(260, vtHost.clientWidth), rowH = 40, H = VT.length * rowH + 6;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Bar chart of VTune CPU time by function" });
    vtHost.insertBefore(svg, vtHost.firstChild);
    var labelW = 0, maxW = W - 48;
    VT.forEach(function (d, i) {
      var y = i * rowH;
      el("text", { class: "dlabel", x: 0, y: y + 13 }, svg).textContent = d.name;
      var w = Math.max(2, (d.p / 35) * (maxW - labelW));
      var bar = el("rect", { x: 0, y: y + 19, width: w, height: 12, rx: 3, fill: d.c, class: "bar" }, svg);
      el("text", { class: "barlabel", x: w + 6, y: y + 29 }, svg).textContent = d.p + "%";
      var hitR = el("rect", { x: 0, y: y, width: W, height: rowH, fill: "transparent" }, svg);
      function show() {
        bar.classList.add("hl");
        var r = svg.getBoundingClientRect(), s = W / r.width;
        vtTip.show('<div class="tt-head">' + d.fn + "</div>" + row(d.c, "CPU time", d.s.toFixed(1) + " s") + row("transparent", "Share", d.p + "%"), Math.min(w, W * 0.5) / s, y / s);
      }
      hitR.addEventListener("pointermove", show);
      hitR.addEventListener("pointerdown", show);
      hitR.addEventListener("pointerleave", function () { bar.classList.remove("hl"); vtTip.hide(); });
    });
  }
  drawVT();

  // ---------- tables ----------
  var tb = document.querySelector("#summary-table tbody");
  IMPL.forEach(function (s) {
    var p26 = s.pts.find(function (p) { return p.n === N26; });
    var pk = peak(s);
    var tr = document.createElement("tr");
    tr.innerHTML = '<td><span class="sw" style="background:' + s.color + '"></span>' + s.name + "</td>" +
      "<td>" + (s.cfg ? p26.k + " " + s.cfg : "n/a") + "</td>" +
      '<td class="num">' + fmtMs(p26.t) + "</td>" +
      '<td class="num">' + fmtX(p26.s) + "×</td>" +
      '<td class="num">' + (s.id === "serial" ? "n/a" : fmtX(pk.s) + "× at " + supN(pk.n)) + "</td>";
    tb.appendChild(tr);
  });

  var raw = document.getElementById("raw-table");
  var cols = [{ h: "Serial", m: serialMap }];
  sortedKeys(G.openmp).forEach(function (k) { cols.push({ h: "OMP " + k, m: G.openmp[k] }); });
  sortedKeys(G.mpi).forEach(function (k) { cols.push({ h: "MPI " + k, m: G.mpi[k] }); });
  sortedKeys(G.cuda).forEach(function (k) { cols.push({ h: "CUDA " + k, m: G.cuda[k] }); });
  cols.push({ h: "JAX CPU", m: JAX.cpu }, { h: "JAX GPU", m: JAX.gpu });
  var html = "<thead><tr><th>n</th>" + cols.map(function (c) { return '<th class="num">' + c.h + "</th>"; }).join("") + "</tr></thead><tbody>";
  NS_LIST.forEach(function (n) {
    html += "<tr><td>" + supN(n) + "</td>" + cols.map(function (c) { return '<td class="num">' + fmtMs(c.m[n]) + "</td>"; }).join("") + "</tr>";
  });
  raw.innerHTML = html + '</tbody><caption style="caption-side:bottom;text-align:left;padding-top:.5rem;color:var(--color-text-faint);font-size:.8rem">Milliseconds per FFT. Source: */results.csv in the repository.</caption>';

  // ---------- equations ----------
  if (window.katex) {
    document.querySelectorAll(".pk-math[data-tex]").forEach(function (m) {
      try { window.katex.render(m.getAttribute("data-tex"), m, { displayMode: true, throwOnError: false }); } catch (e) {}
    });
  }

  // ---------- resize ----------
  var rt;
  var lastW = window.innerWidth;
  window.addEventListener("resize", function () {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    clearTimeout(rt);
    rt = setTimeout(function () { drawSpeed(); drawRun(); drawWeak(); drawExp(); drawVT(); }, 120);
  });
})();
