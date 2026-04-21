// WebSocket streaming client for SignLang recognition.
(function () {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const labelEl = document.getElementById("label");
  const confEl = document.getElementById("confidence");
  const sentenceEl = document.getElementById("sentence");
  const translationEl = document.getElementById("translation");
  const statusEl = document.getElementById("status");
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const resetBtn = document.getElementById("reset");
  const langButtons = document.querySelectorAll(".lang-toggle button");

  let ws = null;
  let stream = null;
  let sendTimer = null;
  let currentTranslations = {};
  let activeLang = "telugu";

  langButtons.forEach((b) => {
    b.addEventListener("click", () => {
      activeLang = b.dataset.lang;
      langButtons.forEach((x) => x.classList.add("inactive"));
      b.classList.remove("inactive");
      renderTranslation();
    });
  });

  function renderTranslation() {
    translationEl.textContent = currentTranslations[activeLang] || "";
  }

  async function start() {
    if (ws) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
    } catch (e) {
      statusEl.textContent = "camera error: " + e.message;
      return;
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/stream`);
    ws.onopen = () => { statusEl.textContent = "connected"; sendTimer = setInterval(sendFrame, 800); };
    ws.onclose = () => { statusEl.textContent = "disconnected"; cleanup(); };
    ws.onerror = () => { statusEl.textContent = "ws error"; };
    ws.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.error) { statusEl.textContent = d.error; return; }
      labelEl.textContent = d.label || "—";
      confEl.textContent = (d.confidence || 0).toFixed(3);
      sentenceEl.textContent = d.sentence || "";
      currentTranslations = d.translations || {};
      renderTranslation();
      drawBoxes(d.detections || []);
      if (window.refreshAnalytics) window.refreshAnalytics();
    };
  }

  function cleanup() {
    if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
    if (ws) { try { ws.close(); } catch (e) {} ws = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  }

  function sendFrame() {
    if (!ws || ws.readyState !== 1 || video.videoWidth === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = 320; canvas.height = 240;
    canvas.getContext("2d").drawImage(video, 0, 0, 320, 240);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    ws.send(JSON.stringify({ image_b64: dataUrl }));
  }

  function drawBoxes(dets) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const sx = overlay.width / 320;
    const sy = overlay.height / 240;
    ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 3; ctx.font = "16px sans-serif"; ctx.fillStyle = "#58a6ff";
    dets.forEach((d) => {
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillText(`${d.label} ${d.confidence.toFixed(2)}`, x1 * sx, y1 * sy - 6);
    });
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", cleanup);
  resetBtn.addEventListener("click", async () => {
    await fetch("/reset-sentence", { method: "POST" });
    sentenceEl.textContent = "";
  });
})();
