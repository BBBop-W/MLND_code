(function () {
  "use strict";
  function phi(s, fleet) { return s <= fleet ? 0 : Math.pow(Math.sqrt(s) - Math.sqrt(fleet), 2); }
  function evaluate(fleet, s1, s2, x1, x2) {
    const f1 = -phi(s1, fleet), f2 = -phi(s2, fleet), f12 = -phi(s1 + s2, fleet);
    const cut12 = f1 * x1 + (f12 - f1) * x2;
    const cut21 = (f12 - f2) * x1 + f2 * x2;
    const chord = s1 + s2 === 0 ? 0 : f12 * (s1 * x1 + s2 * x2) / (s1 + s2);
    return { f1, f2, f12, cut12, cut21, chord, epi: Math.max(cut12, cut21),
      marginalEmpty: f2, marginalAfter: f12 - f1,
      rho1: s1 <= fleet ? 0 : 1 - Math.sqrt(fleet / s1),
      rho12: s1 + s2 <= fleet ? 0 : 1 - Math.sqrt(fleet / (s1 + s2)) };
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { phi, evaluate };
  if (typeof document === "undefined") return;
  const get = id => document.getElementById(id);
  const fmt = x => (Math.abs(x) < 0.0000005 ? 0 : x).toFixed(6);
  function update() {
    const inputs = ["fleet", "s1", "s2", "x1", "x2"];
    const vals = inputs.map(id => Number(get(id).value));
    inputs.forEach((id, i) => get(id + "Value").textContent = vals[i].toFixed(2));
    const [fleet, s1, s2, x1, x2] = vals;
    const r = evaluate(fleet, s1, s2, x1, x2);
    ["f1", "f2", "f12", "cut12", "cut21", "chord", "epi", "marginalEmpty", "marginalAfter"].forEach(id => get(id).textContent = fmt(r[id]));
    ["f1", "f2", "f12"].forEach(id => get(id + "Saving").textContent = fmt(-r[id]));
    get("gain").textContent = fmt(r.epi - r.chord);
    get("chain12").textContent = `θ − C₀ ≥ ${fmt(r.f1)} x₁ ${r.f12 - r.f1 < 0 ? "−" : "+"} ${fmt(Math.abs(r.f12 - r.f1))} x₂`;
    get("chain21").textContent = `θ − C₀ ≥ ${fmt(r.f12 - r.f2)} x₁ ${r.f2 < 0 ? "−" : "+"} ${fmt(Math.abs(r.f2))} x₂`;
    get("ordering").textContent = x1 >= x2 ? "x₁ ≥ x₂：排列 (1, 2) 给出最强割。" : "x₂ ≥ x₁：排列 (2, 1) 给出最强割。";
    const scale = Math.max(Math.abs(r.chord), Math.abs(r.epi), 1e-12);
    get("oldBar").style.width = (Math.abs(r.chord) / scale * 100) + "%";
    get("newBar").style.width = (Math.abs(r.epi) / scale * 100) + "%";
    get("oldBarValue").textContent = fmt(r.chord);
    get("newBarValue").textContent = fmt(r.epi);
    get("availability").textContent = `仅有任务 1 时 ρ* = ${r.rho1.toFixed(4)}；两个任务都加入时 ρ* = ${r.rho12.toFixed(4)}。`;
  }
  ["fleet", "s1", "s2", "x1", "x2"].forEach(id => get(id).addEventListener("input", update));
  get("reset").addEventListener("click", () => { [1,4,4,.5,0].forEach((v,i) => get(["fleet","s1","s2","x1","x2"][i]).value=v); update(); });
  document.querySelectorAll("[data-answer]").forEach(button => button.addEventListener("click", () => {
    get("feedback").textContent = button.dataset.answer === "yes" ? "正确。−1 ≥ −2.343146，带符号的成本边际递减；节约的绝对值同时增大。" : "再看带符号的比较：−1 ≥ −2.343146，所以这两个边际满足次模性。";
  }));
  update();
}());
