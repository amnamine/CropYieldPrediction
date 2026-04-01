(function () {
  const form = document.getElementById("predict-form");
  const errorEl = document.getElementById("form-error");
  const resultPanel = document.getElementById("result-panel");
  const resultValue = document.getElementById("result-value");
  const resultRatio = document.getElementById("result-ratio");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = submitBtn.querySelector(".btn-text");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const resetBtn = document.getElementById("reset-btn");

  const defaults = {
    year: "2010",
    rain: "1200",
    pesticides: "50",
    temp: "18",
  };

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
  }

  function setLoading(loading) {
    submitBtn.disabled = loading;
    btnText.hidden = loading;
    btnSpinner.hidden = !loading;
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    clearError();

    const fd = new FormData(form);
    const area = fd.get("area");
    const item = fd.get("item");
    if (!area || !item) {
      showError("Please select both country and crop.");
      return;
    }

    const payload = {
      area: String(area),
      item: String(item),
      year: Number(fd.get("year")),
      average_rain_fall_mm_per_year: Number(fd.get("average_rain_fall_mm_per_year")),
      pesticides_tonnes: Number(fd.get("pesticides_tonnes")),
      avg_temp: Number(fd.get("avg_temp")),
    };

    if (Object.values(payload).some((v) => typeof v === "number" && Number.isNaN(v))) {
      showError("Check that all numbers are valid.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        showError(data.error || "Prediction failed.");
        return;
      }
      resultValue.textContent = new Intl.NumberFormat().format(data.predicted_yield_hg_per_ha);
      resultRatio.textContent = String(data.rain_temp_ratio);
      resultPanel.hidden = false;
      resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch {
      showError("Network error. Is the Flask server running?");
    } finally {
      setLoading(false);
    }
  });

  resetBtn.addEventListener("click", function () {
    form.reset();
    document.getElementById("year").value = defaults.year;
    document.getElementById("rain").value = defaults.rain;
    document.getElementById("pesticides").value = defaults.pesticides;
    document.getElementById("temp").value = defaults.temp;
    resultPanel.hidden = true;
    clearError();
  });
})();
