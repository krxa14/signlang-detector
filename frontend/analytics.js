// Analytics: pulls /analytics and renders history table + Chart.js bar chart.
(function () {
  const historyEl = document.getElementById("history");
  const chartCtx = document.getElementById("topChart").getContext("2d");
  let chart = null;

  async function refresh() {
    try {
      const r = await fetch("/analytics");
      if (!r.ok) return;
      const d = await r.json();
      historyEl.innerHTML = "";
      (d.recent || []).forEach((p) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${new Date(p.timestamp).toLocaleTimeString()}</td>
                        <td>${p.predicted_label}</td>
                        <td>${p.confidence.toFixed(3)}</td>
                        <td>${p.model_used}</td>`;
        historyEl.appendChild(tr);
      });
      const labels = (d.top_labels || []).map((x) => x.label);
      const counts = (d.top_labels || []).map((x) => x.count);
      if (!chart) {
        chart = new Chart(chartCtx, {
          type: "bar",
          data: { labels, datasets: [{ label: "count", data: counts, backgroundColor: "#58a6ff" }] },
          options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { color: "#e6edf3" } }, x: { ticks: { color: "#e6edf3" } } } },
        });
      } else {
        chart.data.labels = labels;
        chart.data.datasets[0].data = counts;
        chart.update();
      }
    } catch (e) { /* silent */ }
  }

  window.refreshAnalytics = refresh;
  refresh();
  setInterval(refresh, 5000);
})();
