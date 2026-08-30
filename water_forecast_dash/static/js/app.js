const runBtn = document.getElementById("runForecastBtn");
const tankSelect = document.getElementById("tankSelect");
const predictionLengthInput = document.getElementById("predictionLength");
const healthBadge = document.getElementById("healthBadge");
const chartMeta = document.getElementById("chartMeta");
const forecastTableBody = document.querySelector("#forecastTable tbody");
const tableModelViewSelect = document.getElementById("tableModelView");
const retrainModelKeySelect = document.getElementById("retrainModelKey");
const retrainModelBtn = document.getElementById("retrainModelBtn");
const stopRetrainBtn = document.getElementById("stopRetrainBtn");
const retrainStateEl = document.getElementById("retrainState");
const retrainLogEl = document.getElementById("retrainLog");
const waltrTokenInput = document.getElementById("waltrToken");
const locationIdInput = document.getElementById("locationId");
const syncStartDateInput = document.getElementById("syncStartDate");
const syncEndDateInput = document.getElementById("syncEndDate");
const syncDataBtn = document.getElementById("syncDataBtn");
const syncStateEl = document.getElementById("syncState");
const syncLogEl = document.getElementById("syncLog");
const latestActualEl = document.getElementById("latestActual");
const autogluonAvgEl = document.getElementById("autogluonAvg");
const patchtstAvgEl = document.getElementById("patchtstAvg");
const modelRegistryData = document.getElementById("modelRegistryData");

const MODEL_REGISTRY = modelRegistryData ? JSON.parse(modelRegistryData.textContent) : {};

let chartRef = null;
let lastForecastPayload = null;

function selectedModelKeys() {
  const selected = [];
  document.querySelectorAll('input[name="modelKeys"]:checked').forEach((el) => {
    selected.push(el.value);
  });
  return selected;
}

function toFloat(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function fmt(value) {
  const parsed = toFloat(value);
  return parsed === null ? "-" : parsed.toFixed(2);
}

function modelLabel(modelKey) {
  return MODEL_REGISTRY[modelKey]?.label || modelKey;
}

function modelColor(modelKey) {
  if (modelKey === "autogluon") return "#116149";
  if (modelKey === "patchtst") return "#2f68b8";
  return "#5b6b7a";
}

function modelBadgeClass(modelKey) {
  if (modelKey === "autogluon") return "badge-autogluon";
  if (modelKey === "patchtst") return "badge-patchtst";
  return "badge-default";
}

function renderChart(history, forecasts, tankId) {
  const historyLabels = history.map((x) => x.timestamp);
  const historyValues = history.map((x) => x["Outflow in KL"]);

  const datasets = [
    {
      label: "Recent Outflow",
      data: historyValues,
      borderColor: "#2b3b4f",
      backgroundColor: "rgba(43, 59, 79, 0.15)",
      pointRadius: 0,
      borderWidth: 2,
      tension: 0.2,
      fill: false,
    },
  ];

  Object.entries(forecasts).forEach(([modelKey, rows]) => {
    const color = modelColor(modelKey);
    const values = rows.map((x) => x.pred_mean);

    datasets.push({
      label: `${modelLabel(modelKey)} Forecast`,
      data: new Array(historyLabels.length).fill(null).concat(values),
      borderColor: color,
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2,
      borderDash: [8, 4],
      tension: 0.2,
      fill: false,
    });
  });

  const futureLabels = (() => {
    const firstModelRows = Object.values(forecasts)[0] || [];
    return firstModelRows.map((x) => x.timestamp);
  })();

  const labels = historyLabels.concat(futureLabels);

  if (chartRef) {
    chartRef.destroy();
  }

  chartRef = new Chart(document.getElementById("forecastChart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          title: {
            display: true,
            text: "Outflow (KL)",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
      },
    },
  });

  chartMeta.textContent = `${tankId} | Last ${history.length}h history + next ${futureLabels.length}h forecast`;
}

function renderForecastNotes(forecastSources, warnings) {
  const sourceParts = Object.entries(forecastSources || {}).map(
    ([modelKey, source]) => `${modelLabel(modelKey)}: ${source}`
  );

  if (sourceParts.length) {
    const sourceText = `Sources -> ${sourceParts.join(" | ")}`;
    chartMeta.textContent = `${chartMeta.textContent} | ${sourceText}`;
  }

  if (warnings && warnings.length) {
    chartMeta.textContent = `${chartMeta.textContent} | Warning: ${warnings.join("; ")}`;
  }
}

function renderForecastTable(forecasts, preferredModelKey) {
  const availableModelKeys = Object.keys(forecasts || {}).filter(
    (k) => Array.isArray(forecasts[k]) && forecasts[k].length
  );

  if (!availableModelKeys.length) {
    forecastTableBody.innerHTML = '<tr><td colspan="6">No forecast rows available.</td></tr>';
    return;
  }

  if (tableModelViewSelect) {
    const currentVal = tableModelViewSelect.value;
    tableModelViewSelect.innerHTML = availableModelKeys
      .map((k) => `<option value="${k}">${modelLabel(k)}</option>`)
      .join("");
    tableModelViewSelect.value = availableModelKeys.includes(currentVal)
      ? currentVal
      : (preferredModelKey && availableModelKeys.includes(preferredModelKey) ? preferredModelKey : availableModelKeys[0]);
  }

  const selectedKey = tableModelViewSelect?.value && availableModelKeys.includes(tableModelViewSelect.value)
    ? tableModelViewSelect.value
    : availableModelKeys[0];
  const selectedRows = forecasts[selectedKey] || [];
  const badgeClass = modelBadgeClass(selectedKey);

  forecastTableBody.innerHTML = selectedRows
    .map((r, idx) => {
      const p10 = fmt(r.pred_p10);
      const p90 = fmt(r.pred_p90);
      const rangeText = p10 !== "-" && p90 !== "-" ? `${p10} - ${p90}` : "-";
      return `
      <tr>
        <td><span class="hour-pill">H+${idx + 1}</span></td>
        <td class="forecast-ts">${r.timestamp}</td>
        <td class="forecast-main">
          <span class="model-badge ${badgeClass}">${modelLabel(selectedKey)}</span>
          <strong>${fmt(r.pred_mean)}</strong>
        </td>
        <td class="num-col">${p10}</td>
        <td class="num-col">${p90}</td>
        <td class="forecast-range">${rangeText}</td>
      </tr>
    `;
    })
    .join("");
}

function updateMetricCards(history, forecasts) {
  const latestHistory = history[history.length - 1];
  latestActualEl.textContent = latestHistory ? `${fmt(latestHistory["Outflow in KL"])} KL` : "-";

  const avgByModel = {};
  Object.entries(forecasts).forEach(([modelKey, rows]) => {
    const vals = rows.map((x) => toFloat(x.pred_mean)).filter((x) => x !== null);
    if (vals.length) {
      avgByModel[modelKey] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
  });

  autogluonAvgEl.textContent = avgByModel.autogluon !== undefined ? `${fmt(avgByModel.autogluon)} KL` : "-";
  patchtstAvgEl.textContent = avgByModel.patchtst !== undefined ? `${fmt(avgByModel.patchtst)} KL` : "-";
}

function showError(message) {
  chartMeta.innerHTML = `<span class="error-text">${message}</span>`;
}

function renderSyncSummary(payload) {
  const sync = payload?.sync || {};
  const lines = [];
  const downloadedTotal = Number(sync.downloaded_total || 0);
  const errorsTotal = Number(sync.errors_total || 0);

  if (downloadedTotal === 0 && errorsTotal === 0) {
    lines.push("status=Data is up to date (no new files downloaded)");
  } else {
    lines.push("status=Sync completed with new data");
  }

  lines.push(`location_id=${sync.location_id}`);
  lines.push(`date_range=${sync.start_date} -> ${sync.end_date}`);
  lines.push(`tanks=${sync.tanks_total}`);
  lines.push(`downloaded=${sync.downloaded_total}, skipped_existing=${sync.skipped_existing_total}, no_data_days=${sync.no_data_days_total}, errors=${sync.errors_total}`);

  if (Array.isArray(sync.downloaded_files_sample) && sync.downloaded_files_sample.length) {
    lines.push("--- downloaded files (sample) ---");
    sync.downloaded_files_sample.slice(0, 80).forEach((p) => lines.push(p));
  }

  if (Array.isArray(sync.per_tank) && sync.per_tank.length) {
    lines.push("--- per tank ---");
    sync.per_tank.slice(0, 30).forEach((t) => {
      lines.push(
        `${t.dataset_dir}: start=${t.start_date}, end=${t.end_date}, downloaded=${t.downloaded}, skipped=${t.skipped_existing}, no_data=${t.no_data_days}, errors=${t.errors}`
      );
    });
  }

  syncLogEl.textContent = lines.join("\n");
}

async function syncLatestData() {
  const token = (waltrTokenInput?.value || "").trim();
  if (!token) {
    if (syncStateEl) syncStateEl.textContent = "Missing token";
    return;
  }

  const locationId = (locationIdInput?.value || "").trim();
  const startDate = (syncStartDateInput?.value || "2025-01-01").trim();
  const endDate = (syncEndDateInput?.value || new Date().toISOString().slice(0, 10)).trim();

  syncDataBtn.disabled = true;
  syncDataBtn.textContent = "Syncing...";
  if (syncStateEl) syncStateEl.textContent = "Running";

  const startedAt = new Date();
  if (syncLogEl) {
    const locationLabel = locationId || "<default>";
    syncLogEl.textContent = [
      "Sync started...",
      `started_at=${startedAt.toISOString()}`,
      `location_id=${locationLabel}`,
      `date_range=${startDate} -> ${endDate}`,
      "Waiting for server response. Large ranges can take a few minutes.",
    ].join("\n");
  }

  const heartbeatTimer = setInterval(() => {
    if (!syncLogEl) return;
    const elapsedSec = Math.max(1, Math.round((Date.now() - startedAt.getTime()) / 1000));
    syncLogEl.textContent += `\nstill_running=true elapsed_seconds=${elapsedSec}`;
  }, 5000);

  try {
    const body = {
      token,
      start_date: startDate,
      end_date: endDate,
    };
    if (locationId) {
      body.location_id = locationId;
    }

    const res = await fetch("/api/sync-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const raw = await res.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch (_err) {
      data = { error: raw || `Sync failed with HTTP ${res.status}` };
    }

    if (!res.ok) {
      throw new Error(data.error || "Sync failed.");
    }

    const downloadedTotal = Number(data?.sync?.downloaded_total || 0);
    const errorsTotal = Number(data?.sync?.errors_total || 0);
    const stateLabel = downloadedTotal === 0 && errorsTotal === 0
      ? "Completed. Data is up to date. Reloading..."
      : "Completed. Reloading...";
    if (syncStateEl) syncStateEl.textContent = stateLabel;
    renderSyncSummary(data);
    await refreshTankOptions();
    await refreshHealth();
    setTimeout(() => window.location.reload(), 1200);
  } catch (err) {
    if (syncStateEl) syncStateEl.textContent = `Failed: ${err.message}`;
    if (syncLogEl) {
      const lines = [
        "Sync failed.",
        `reason=${err.message}`,
        "If server was restarted, refresh the page and try again.",
      ];
      syncLogEl.textContent = lines.join("\n");
    }
  } finally {
    clearInterval(heartbeatTimer);
    syncDataBtn.disabled = false;
    syncDataBtn.textContent = "Sync Latest Data";
  }
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    const visibleModelKeys = Object.keys(MODEL_REGISTRY || {});
    const availableCount = visibleModelKeys.filter((key) => data.models?.[key]?.available).length;
    const totalCount = visibleModelKeys.length;
    healthBadge.textContent = `Models ready: ${availableCount}/${totalCount}`;
  } catch (_err) {
    healthBadge.textContent = "Unable to read model status";
  }
}

async function runForecast() {
  const tankId = tankSelect.value;
  const predictionLength = Number(predictionLengthInput.value || 24);
  const modelKeys = selectedModelKeys();

  if (!tankId) {
    showError("Select a tank to run forecast.");
    return;
  }

  if (!modelKeys.length) {
    showError("Select at least one model.");
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Forecasting...";

  try {
    const res = await fetch("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tank_id: tankId,
        prediction_length: predictionLength,
        model_keys: modelKeys,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Forecast request failed.");
    }

    lastForecastPayload = data;
    renderChart(data.history, data.forecasts, data.tank_id);
    renderForecastTable(data.forecasts, tableModelViewSelect?.value);
    updateMetricCards(data.history, data.forecasts);
    renderForecastNotes(data.forecast_sources, data.warnings);
  } catch (err) {
    showError(err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run 24h Forecast";
  }
}

function renderRetrainStatus(payload) {
  const retrain = payload?.retrain || {};
  retrainStateEl.textContent = retrain.running
    ? `Running ${retrain.model_key ? `(${retrain.model_key}) ` : ""}(pid=${retrain.pid})`
    : `Idle${retrain.last_exit_code !== null && retrain.last_exit_code !== undefined ? ` | last_exit=${retrain.last_exit_code}` : ""}`;

  retrainModelBtn.disabled = Boolean(retrain.running);
  stopRetrainBtn.disabled = !Boolean(retrain.running);
  if (retrain.running && retrain.model_key && retrainModelKeySelect) {
    retrainModelKeySelect.value = retrain.model_key;
  }

  const lines = Array.isArray(retrain.log_tail) ? retrain.log_tail : [];
  retrainLogEl.textContent = lines.length ? lines.join("\n") : "No retrain logs yet.";
}

async function refreshRetrainStatus() {
  try {
    const res = await fetch("/api/retrain-status");
    if (!res.ok) {
      return;
    }
    const payload = await res.json();
    renderRetrainStatus(payload);
  } catch (_err) {
    // No-op: retrain status is best-effort.
  }
}

async function startRetrainModel() {
  retrainModelBtn.disabled = true;
  retrainModelBtn.textContent = "Starting...";

  try {
    const modelKey = retrainModelKeySelect ? retrainModelKeySelect.value : "patchtst";
    const res = await fetch("/api/retrain-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_key: modelKey,
        prediction_length: Number(predictionLengthInput.value || 24),
        time_limit: 1800,
        min_history_hours: 24 * 14,
        training_profile: "baseline",
        num_val_windows: 1,
        refit_full: true,
        enable_deep_models: false,
        max_epochs: modelKey === "patchtst" ? 100 : 200,
        presets: "high_quality",
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Could not start model retraining.");
    }

    renderRetrainStatus(data);
  } catch (err) {
    showError(err.message);
  } finally {
    retrainModelBtn.disabled = false;
    retrainModelBtn.textContent = "Retrain Forecast Model";
  }
}

async function stopRetrainModel() {
  stopRetrainBtn.disabled = true;
  stopRetrainBtn.textContent = "Stopping...";
  try {
    const res = await fetch("/api/retrain-stop", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Could not stop training.");
    }
    renderRetrainStatus(data);
  } catch (err) {
    showError(err.message);
  } finally {
    stopRetrainBtn.textContent = "Stop Training";
  }
}

async function refreshTankOptions() {
  try {
    const current = tankSelect.value;
    const res = await fetch("/api/tanks");
    if (!res.ok) {
      return;
    }

    const data = await res.json();
    const tanks = Array.isArray(data.tanks) ? data.tanks : [];
    if (!tanks.length) {
      return;
    }

    const existing = new Set(Array.from(tankSelect.options).map((opt) => opt.value));
    const incoming = new Set(tanks);

    let changed = false;
    if (existing.size !== incoming.size) {
      changed = true;
    } else {
      for (const t of incoming) {
        if (!existing.has(t)) {
          changed = true;
          break;
        }
      }
    }

    if (!changed) {
      return;
    }

    tankSelect.innerHTML = tanks
      .map((tank) => `<option value="${tank}">${tank}</option>`)
      .join("");

    if (incoming.has(current)) {
      tankSelect.value = current;
    }
  } catch (_err) {
    // No-op: tank refresh is best-effort.
  }
}

runBtn.addEventListener("click", runForecast);
retrainModelBtn.addEventListener("click", startRetrainModel);
stopRetrainBtn.addEventListener("click", stopRetrainModel);
syncDataBtn.addEventListener("click", syncLatestData);
tableModelViewSelect?.addEventListener("change", () => {
  if (!lastForecastPayload) return;
  renderForecastTable(lastForecastPayload.forecasts, tableModelViewSelect.value);
});
window.addEventListener("load", async () => {
  if (syncEndDateInput && !syncEndDateInput.value) {
    syncEndDateInput.value = new Date().toISOString().slice(0, 10);
  }
  await refreshHealth();
  await refreshRetrainStatus();
  await refreshTankOptions();
  await runForecast();
  setInterval(async () => {
    await refreshTankOptions();
    await refreshHealth();
    await refreshRetrainStatus();
  }, 60000);
});
