/* Waltr Forecast Dock — content script.
 *
 * Injects a collapsible right-edge panel over app.waltr.in showing Chronos-2 water-demand
 * forecasts for the 24 PES University RR tanks.
 *
 * Two design constraints drive this file:
 *
 *  1. Everything lives inside a **shadow root**. Waltr's stylesheet cannot reach in and this
 *     file's styles cannot leak out, so the host app renders exactly as it did before. It also
 *     means a fixed-position host works whether Waltr paints to the DOM or, as a Flutter Web
 *     build would, to a <canvas>.
 *
 *  2. Tank selection is **self-contained**. A canvas-rendered host exposes no DOM text to scrape,
 *     so the dock ships its own searchable tank list rather than depending on reading Waltr's
 *     state. URL-based detection is a best-effort enhancement, never a dependency.
 */

(() => {
  "use strict";

  const HOST_ID = "waltr-forecast-dock-host";
  if (document.getElementById(HOST_ID)) return;

  const HORIZONS = [
    { h: 6,   label: "6h"  },
    { h: 12,  label: "12h" },
    { h: 24,  label: "1d"  },
    { h: 48,  label: "2d"  },
    { h: 72,  label: "3d"  },
    { h: 168, label: "7d"  },
  ];

  const store = {
    get: (k, d) => new Promise((res) => {
      try { chrome.storage.local.get([k], (v) => res(v?.[k] ?? d)); }
      catch { res(d); }
    }),
    set: (k, v) => { try { chrome.storage.local.set({ [k]: v }); } catch { /* no-op */ } },
  };

  const state = { bundle: null, tankId: null, horizon: 24, collapsed: false, pickerOpen: false };

  // ── SVG chart ───────────────────────────────────────────────────────────────
  // Hand-rolled rather than pulling a charting library into a content script.
  // Geometry follows the Waltr mockup: history solid, forecast over a hatched future region,
  // p10–p90 band, dashed "now" divider.
  function chart(series, horizon) {
    const W = 372, H = 150;
    const padL = 30, padR = 8, padT = 10, padB = 20;
    const iw = W - padL - padR, ih = H - padT - padB;

    const pts = series.history.concat(series.forecast);
    const lo = Math.min(0, ...series.forecast.map((d) => d.p10));
    const hi = Math.max(
      ...pts.map((d) => (d.actual ?? d.pred ?? 0)),
      ...series.forecast.map((d) => d.p90),
      0.001
    );
    const span = hi - lo || 1;
    const n = pts.length - 1 || 1;
    const X = (i) => padL + (i / n) * iw;
    const Y = (v) => padT + ih - ((v - lo) / span) * ih;

    const nowIdx = series.history.length - 1;
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

    // Axis labels adapt to the range: low-flow tanks run at ~0.05 KL/h, where a fixed one
    // decimal collapses every gridline to "0.0".
    const decimals = (() => {
      const m = Math.max(Math.abs(hi), Math.abs(lo));
      if (m >= 10) return 0;
      if (m >= 1) return 1;
      if (m >= 0.1) return 2;
      return 3;
    })();

    // Gridlines + y labels
    let grid = "";
    for (let g = 0; g <= 3; g++) {
      const v = lo + (span * g) / 3, y = Y(v);
      grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}"
                 stroke="var(--hairline-2)" stroke-width="1"/>
               <text x="${padL - 5}" y="${(y + 3).toFixed(1)}" text-anchor="end"
                 font-size="8" fill="var(--ink-400)" font-family="var(--font-mono)"
                 >${v.toFixed(decimals)}</text>`;
    }

    const histLine = series.history
      .map((d, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(d.actual).toFixed(1)}`).join("");
    const fcLine = series.forecast
      .map((d, i) => `${i ? "L" : "M"}${X(nowIdx + 1 + i).toFixed(1)},${Y(d.pred).toFixed(1)}`)
      .join("");

    const bandTop = series.forecast
      .map((d, i) => `${i ? "L" : "M"}${X(nowIdx + 1 + i).toFixed(1)},${Y(d.p90).toFixed(1)}`).join("");
    const bandBot = series.forecast
      .map((d, i) => `${X(nowIdx + 1 + i).toFixed(1)},${Y(d.p10).toFixed(1)}`)
      .reverse().join("L");
    const band = series.forecast.length ? `${bandTop}L${bandBot}Z` : "";

    const histArea = series.history.length
      ? `${histLine}L${X(nowIdx).toFixed(1)},${Y(lo).toFixed(1)}L${X(0).toFixed(1)},${Y(lo).toFixed(1)}Z`
      : "";
    const nowX = X(nowIdx).toFixed(1);

    return `
    <svg viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Forecast for ${esc(series.tank)} over the next ${horizon} hours">
      <defs>
        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2B8BF5" stop-opacity=".18"/>
          <stop offset="100%" stop-color="#2B8BF5" stop-opacity="0"/>
        </linearGradient>
        <pattern id="fhatch" width="6" height="6" patternTransform="rotate(45)"
                 patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="#F6F8FB"/>
          <line x1="0" y1="0" x2="0" y2="6" stroke="#EEF2F7" stroke-width="3"/>
        </pattern>
      </defs>
      <rect x="${nowX}" y="${padT}" width="${(W - padR - parseFloat(nowX)).toFixed(1)}"
            height="${ih}" fill="url(#fhatch)"/>
      ${grid}
      <path d="${histArea}" fill="url(#ag)"/>
      <path d="${band}" fill="#2B8BF5" fill-opacity=".13"/>
      <path d="${histLine}" fill="none" stroke="#0E1726" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round"/>
      <path d="${fcLine}" fill="none" stroke="#2B8BF5" stroke-width="1.8"
            stroke-linejoin="round" stroke-linecap="round"/>
      <line x1="${nowX}" y1="${padT}" x2="${nowX}" y2="${padT + ih}"
            stroke="#0E1726" stroke-width="1" stroke-dasharray="3 3" opacity=".45"/>
      <text x="${nowX}" y="${padT - 2}" text-anchor="middle" font-size="8"
            fill="var(--ink-500)" font-family="var(--font-mono)">now</text>
    </svg>`;
  }

  // ── Data shaping ────────────────────────────────────────────────────────────
  function seriesFor(tankId, horizon) {
    const t = state.bundle.tanks[tankId];
    const fc = t.forecasts[String(horizon)] || [];
    // Show a history window proportional to the horizon, capped so short horizons stay readable.
    const histLen = Math.min(t.history.length, Math.max(24, Math.round(horizon * 1.5)));
    return {
      tank: t.name,
      history: t.history.slice(-histLen).map((v, i) => ({ actual: v, i })),
      forecast: fc,
    };
  }

  function totals(tankId, horizon) {
    const fc = state.bundle.tanks[tankId].forecasts[String(horizon)] || [];
    const sum = fc.reduce((a, d) => a + d.pred, 0);
    const peak = fc.reduce((a, d) => (d.pred > a.pred ? d : a), { pred: -1, t: "" });
    return { sum, peak };
  }

  function hoursToEmpty(tankId, horizon) {
    const t = state.bundle.tanks[tankId];
    const fc = t.forecasts[String(horizon)] || [];
    if (t.level_kl == null || !fc.length) return null;
    let left = t.level_kl;
    for (let i = 0; i < fc.length; i++) {
      left -= fc[i].pred;
      if (left <= 0) return i + 1;
    }
    return null;
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  function render(root) {
    const b = state.bundle;
    const tank = b.tanks[state.tankId];
    const { sum, peak } = totals(state.tankId, state.horizon);
    const hLabel = HORIZONS.find((x) => x.h === state.horizon).label;
    const empty = hoursToEmpty(state.tankId, state.horizon);
    const acc = tank.accuracy?.[String(state.horizon)];

    const tankRows = Object.entries(b.tanks)
      .sort((a, c) => c[1].mean_kl - a[1].mean_kl)
      .map(([id, t]) => {
        const cls = t.trust === "dead" ? "bad" : t.trust === "degraded" ? "warn" : "";
        const pct = t.level_pct == null ? "--" : `${Math.round(t.level_pct)}%`;
        return `<div class="picker-item ${id === state.tankId ? "active" : ""}" data-tank="${id}">
                  <span class="dot ${cls}"></span><span>${t.name}</span>
                  <span class="lvl">${pct}</span>
                </div>`;
      }).join("");

    let alert = "";
    if (tank.trust === "dead") {
      alert = `<span class="chip red">Sensor reports no flow — forecast not meaningful</span>`;
    } else if (empty !== null) {
      alert = `<span class="chip amber">Refill needed in ~${empty}h at forecast demand</span>`;
    } else {
      alert = `<span class="chip green">Level holds through the next ${hLabel}</span>`;
    }

    root.innerHTML = `
    <div class="dock ${state.collapsed ? "collapsed" : ""}">
      <div class="rail" data-act="expand">
        <span class="vlabel">Forecast</span>
      </div>
      <div class="hd">
        <div class="hd-mark">W</div>
        <div>
          <div class="hd-title">Forecast</div>
          <div class="hd-sub">PES University RR</div>
        </div>
        <div class="hd-spacer"></div>
        <button class="iconbtn" data-act="collapse" title="Collapse panel">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 18l6-6-6-6"/></svg>
        </button>
      </div>

      <div class="dock-body">
        <div class="sec">
          <div class="sec-label">Tank</div>
          <div class="picker">
            <button class="picker-btn" data-act="toggle-picker">
              <span class="name">${tank.name}</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M6 9l6 6 6-6"/></svg>
            </button>
            <div class="picker-pop" ${state.pickerOpen ? "" : "hidden"}>
              <input class="picker-search" type="text" placeholder="Search tanks…" />
              <div class="picker-list">${tankRows}</div>
            </div>
          </div>
        </div>

        <div class="sec">
          <div class="sec-label">Predicted demand · next ${hLabel}</div>
          <div class="headline">
            <span class="val">${sum.toFixed(1)}</span><span class="unit">KL</span>
          </div>
          <div class="sub">
            Peak ${peak.pred >= 0 ? peak.pred.toFixed(2) : "--"} KL/h${peak.t ? ` at ${peak.t}` : ""}
          </div>
          <div class="chartwrap">${chart(seriesFor(state.tankId, state.horizon), state.horizon)}</div>
        </div>

        <div class="sec">
          <div class="sec-label">Horizon</div>
          <div class="segment">
            ${HORIZONS.map((x) =>
              `<button data-h="${x.h}" class="${x.h === state.horizon ? "on" : ""}">${x.label}</button>`
            ).join("")}
          </div>
        </div>

        <div class="sec">${alert}</div>

        <div class="sec">
          <div class="sec-label">Detail</div>
          <div class="kv"><span class="k">Current level</span>
            <span class="v">${tank.level_kl == null ? "--" : tank.level_kl.toFixed(1) + " KL"}</span></div>
          <div class="kv"><span class="k">Avg demand</span>
            <span class="v">${tank.mean_kl.toFixed(2)} KL/h</span></div>
          <div class="kv"><span class="k">Sensor health</span>
            <span class="v">${tank.trust}</span></div>
          ${acc && acc.mase != null ? `<div class="kv"><span class="k">Model MASE @ ${hLabel}</span>
            <span class="v">${acc.mase.toFixed(2)}</span></div>` : ""}
        </div>
      </div>

      <div class="foot">
        <span>${b.model}</span>
        <span class="mono">${b.generated_at}</span>
      </div>
    </div>`;

    wire(root);
  }

  function wire(root) {
    const dock = root.querySelector(".dock");

    root.querySelector('[data-act="collapse"]')?.addEventListener("click", () => {
      state.collapsed = true; store.set("collapsed", true); dock.classList.add("collapsed");
    });
    root.querySelector('[data-act="expand"]')?.addEventListener("click", () => {
      state.collapsed = false; store.set("collapsed", false); dock.classList.remove("collapsed");
    });
    root.querySelector('[data-act="toggle-picker"]')?.addEventListener("click", (e) => {
      e.stopPropagation();
      state.pickerOpen = !state.pickerOpen;
      render(root);
      if (state.pickerOpen) root.querySelector(".picker-search")?.focus();
    });

    root.querySelectorAll(".segment button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.horizon = Number(btn.dataset.h);
        store.set("horizon", state.horizon);
        render(root);
      });
    });

    root.querySelectorAll(".picker-item").forEach((item) => {
      item.addEventListener("click", () => {
        state.tankId = item.dataset.tank;
        state.pickerOpen = false;
        store.set("tankId", state.tankId);
        render(root);
      });
    });

    const search = root.querySelector(".picker-search");
    search?.addEventListener("input", () => {
      const q = search.value.toLowerCase();
      root.querySelectorAll(".picker-item").forEach((item) => {
        const name = item.textContent.toLowerCase();
        item.style.display = name.includes(q) ? "" : "none";
      });
    });
    search?.addEventListener("click", (e) => e.stopPropagation());

    if (!wire._outsideBound) {
      document.addEventListener("click", (ev) => {
        if (!state.pickerOpen) return;
        if (!ev.composedPath().includes(root.host)) {
          state.pickerOpen = false;
          render(root);
        }
      });
      wire._outsideBound = true;
    }
  }

  // Best-effort: if the URL names a tank we know, preselect it. Never required.
  function tankFromUrl(bundle) {
    const slug = decodeURIComponent(location.href).toUpperCase().replace(/[^A-Z0-9]/g, "");
    for (const id of Object.keys(bundle.tanks)) {
      if (slug.includes(id.replace(/[^A-Z0-9]/g, ""))) return id;
    }
    return null;
  }

  async function start() {
    const host = document.createElement("div");
    host.id = HOST_ID;
    document.documentElement.appendChild(host);
    const shadow = host.attachShadow({ mode: "open" });

    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = chrome.runtime.getURL("dock.css");
    shadow.appendChild(link);

    const mount = document.createElement("div");
    shadow.appendChild(mount);
    mount.host = host;

    const res = await fetch(chrome.runtime.getURL("forecast_bundle.json"));
    state.bundle = await res.json();

    const ids = Object.keys(state.bundle.tanks);
    const saved = await store.get("tankId", null);
    state.tankId = (saved && ids.includes(saved))
      ? saved
      : (tankFromUrl(state.bundle)
         || ids.sort((a, b) => state.bundle.tanks[b].mean_kl - state.bundle.tanks[a].mean_kl)[0]);
    state.horizon = Number(await store.get("horizon", 24));
    state.collapsed = Boolean(await store.get("collapsed", false));

    render(mount);
  }

  start().catch((err) => console.error("[waltr-dock]", err));
})();
