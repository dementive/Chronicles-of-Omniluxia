/* Small progressive-enhancement helpers for indexes and data compendia. */
(function () {
  "use strict";

  function wireSearch(input, items, count, emptyText) {
    if (!input || !items.length) return;
    input.addEventListener("input", function () {
      var query = input.value.trim().toLowerCase();
      var visible = 0;
      items.forEach(function (item) {
        var matches = !query || item.textContent.toLowerCase().indexOf(query) !== -1;
        item.hidden = !matches;
        if (matches) visible += 1;
      });
      if (count) {
        var plural = visible === 1 ? emptyText : (emptyText === "entry" ? "entries" : emptyText + "s");
        count.textContent = visible.toLocaleString() + " " + plural;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var menu = document.querySelector(".nav-menu");
    if (menu && window.matchMedia("(max-width: 1300px)").matches) menu.open = false;
    wireSearch(
      document.querySelector("[data-index-search]"),
      Array.prototype.slice.call(document.querySelectorAll("[data-index-item]")),
      document.querySelector("[data-index-count]"),
      "page"
    );
    wireSearch(
      document.querySelector("[data-entry-search]"),
      Array.prototype.slice.call(document.querySelectorAll(".data-entry")),
      document.querySelector("[data-entry-count]"),
      "entry"
    );
  });
}());
