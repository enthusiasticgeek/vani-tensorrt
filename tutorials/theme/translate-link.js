// "Translate this page" -- a small language-picker dropdown in the
// menu bar, rather than a single button that auto-guesses a target
// language from the visitor's browser locale. Auto-guessing broke
// down for a real, common case: a multilingual reader (e.g. fluent in
// English and Hindi, browser locale set to one of those) who
// specifically wants a THIRD language (e.g. Marathi) had no way to
// override the guess. A `<select>` gives an explicit, always-visible
// choice instead. Opens Google Translate's <host>.translate.goog
// proxy view of the current page in a new tab -- see the two fixes
// below for why that specific mechanism, not translate.google.com's
// own redirect page or an embedded widget.
//
// History (all found by testing the real deployed site, not by
// reasoning from docs):
//   1. mdBook renamed #menu-bar to #mdbook-menu-bar around 0.5.x; this
//      project's CI pins an older mdBook that still uses the
//      unprefixed ID, so a selector matching only the new one never
//      found the menu bar on the real site.
//   2. Chrome for Android specially intercepts links to
//      translate.google.com/translate and hands them to its own
//      native translate-infobar flow instead of navigating normally --
//      that native flow failed independently of the link's own
//      correctness. Fixed by linking straight to the actual
//      <host>.translate.goog proxy URL (what translate.google.com's
//      redirect would have sent Chrome to anyway), which is just an
//      ordinary external link from Chrome's point of view.
//   3. translate.goog returns a hard 400 (not a harmless no-op) when
//      the target language equals the source language -- this list
//      deliberately excludes English (the site's own source language)
//      so that case can't occur by construction.
(function () {
  var rightButtons = document.querySelector(
    "#mdbook-menu-bar .right-buttons, #menu-bar .right-buttons"
  );
  if (!rightButtons) {
    return;
  }

  var SOURCE_LANG = "en"; // every page on this site is authored in English

  // Curated, not exhaustive -- Indian languages first (this project's
  // own dialect list overlaps heavily), then other major languages.
  // Google Translate supports many more; add here if a reader asks
  // for one that's missing.
  var LANGUAGES = [
    ["", "\u{1F310} Translate to…"],
    ["hi", "हिन्दी (Hindi)"],
    ["mr", "मराठी (Marathi)"],
    ["bn", "বাংলা (Bengali)"],
    ["ta", "தமிழ் (Tamil)"],
    ["te", "తెలుగు (Telugu)"],
    ["gu", "ગુજરતી (Gujarati)"],
    ["pa", "ਪੰਜਾਬੀ (Punjabi)"],
    ["kn", "ಕನ್ನಡ (Kannada)"],
    ["ml", "മലയാളം (Malayalam)"],
    ["or", "ଓଡ଼ିଆ (Odia)"],
    ["ur", "اردو (Urdu)"],
    ["ne", "नेपाली (Nepali)"],
    ["si", "සිංහල (Sinhala)"],
    ["es", "Español (Spanish)"],
    ["fr", "Français (French)"],
    ["de", "Deutsch (German)"],
    ["pt", "Português (Portuguese)"],
    ["ru", "Русский (Russian)"],
    ["zh-CN", "中文 (Chinese, Simplified)"],
    ["ja", "日本語 (Japanese)"],
    ["ko", "한국어 (Korean)"],
    ["ar", "العربية (Arabic)"],
    ["it", "Italiano (Italian)"],
    ["tr", "Türkçe (Turkish)"],
    ["vi", "Tiếng Việt (Vietnamese)"],
    ["th", "ไทย (Thai)"],
    ["id", "Bahasa Indonesia (Indonesian)"],
    ["nl", "Nederlands (Dutch)"],
    ["pl", "Polski (Polish)"],
    ["sw", "Kiswahili (Swahili)"],
    ["fa", "فارسی (Persian)"],
    ["he", "עברית (Hebrew)"],
    ["el", "Ελληνικά (Greek)"],
    ["uk", "Українська (Ukrainian)"]
  ];

  function translateUrlFor(targetLang) {
    var host = window.location.hostname.replace(/-/g, "--").replace(/\./g, "-");
    return (
      "https://" +
      host +
      ".translate.goog" +
      window.location.pathname +
      "?_x_tr_sl=" +
      SOURCE_LANG +
      "&_x_tr_tl=" +
      encodeURIComponent(targetLang) +
      "&_x_tr_hl=" +
      encodeURIComponent(targetLang) +
      "&_x_tr_pto=wapp"
    );
  }

  var select = document.createElement("select");
  select.id = "translate-button";
  select.title = "Translate this page";
  select.setAttribute("aria-label", "Translate this page");
  select.className = "fa-svg";
  // Blend into the menu bar's icon-button row -- a bare <select>
  // otherwise renders with browser-default chrome that looks out of
  // place next to the SVG icon buttons.
  select.style.background = "transparent";
  select.style.border = "none";
  select.style.color = "inherit";
  select.style.font = "inherit";
  select.style.cursor = "pointer";
  select.style.maxWidth = "2.2em";

  LANGUAGES.forEach(function (entry) {
    var opt = document.createElement("option");
    opt.value = entry[0];
    opt.textContent = entry[1];
    select.appendChild(opt);
  });

  // Pre-highlight the visitor's own browser language if it's one of
  // the offered (non-English) options -- saves a scroll for the
  // common case, but never auto-navigates on its own; only an actual
  // selection (the `change` event) does.
  var browserLang = ((navigator.language || navigator.userLanguage || "").split("-")[0] || "").toLowerCase();
  if (LANGUAGES.some(function (entry) { return entry[0] === browserLang; })) {
    select.value = browserLang;
  }

  select.addEventListener("change", function () {
    if (!select.value) {
      return;
    }
    window.open(translateUrlFor(select.value), "_blank", "noopener,noreferrer");
    select.value = ""; // reset to the placeholder so it can be used again
  });

  rightButtons.insertBefore(select, rightButtons.firstChild);
})();
