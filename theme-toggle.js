/* =====================
   LIGHT / DARK THEME TOGGLE
   ===================== */

const themeBtn = document.getElementById("themeButton");

// Load saved preference
if (localStorage.getItem("theme") === "dark") {
  document.body.classList.add("dark");
  themeBtn.textContent = "☀️ Light";
}

// Toggle Action
themeBtn.addEventListener("click", () => {
  document.body.classList.toggle("dark");

  let isDark = document.body.classList.contains("dark");
  themeBtn.textContent = isDark ? "☀️ Light" : "🌙 Dark";

  localStorage.setItem("theme", isDark ? "dark" : "light");
});
