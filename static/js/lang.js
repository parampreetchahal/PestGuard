function selectLang(lang) {
    const modal = document.getElementById("langModal");

    localStorage.setItem("lang", lang);
    localStorage.setItem("langSelected", "true");

    applyLanguage(lang);

    // ✅ HIDE MODAL IMMEDIATELY
    if (modal) modal.style.display = "none";

    // optional (menu)
    const menu = document.getElementById("langMenu");
    if (menu) menu.style.display = "none";
}

function applyLanguage(lang) {
    // TEXT
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (translations[lang] && translations[lang][key]) {
            el.innerHTML = translations[lang][key];
        }
    });

    // 🔥 PLACEHOLDERS
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
        const key = el.getAttribute("data-i18n-placeholder");
        if (translations[lang] && translations[lang][key]) {
            el.placeholder = translations[lang][key];
        }
    });

    // TITLE FIX
    const titleEl = document.querySelector("title[data-i18n]");
    if (titleEl) {
        const key = titleEl.getAttribute("data-i18n");
        document.title = translations[lang][key];
    }

    document.getElementById("htmlTag").setAttribute("lang", lang);
}

document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("langModal");

    let lang = localStorage.getItem("lang") || "en";

    if (modal) {
        if (!localStorage.getItem("langSelected")) {
            modal.style.display = "flex";
        } else {
            modal.style.display = "none";
        }
    }

    applyLanguage(lang);
};

function toggleLangMenu() {
    const menu = document.getElementById("langMenu");
    menu.style.display = menu.style.display === "block" ? "none" : "block";
}
