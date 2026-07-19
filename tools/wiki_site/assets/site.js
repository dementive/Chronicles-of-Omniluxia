(() => {
  const input = document.querySelector('#wiki-search');
  if (!input) return;

  const rows = [...document.querySelectorAll('.idx-list li')];
  const groups = [...document.querySelectorAll('[data-category-group]')];
  const buttons = [...document.querySelectorAll('[data-filter]')];
  const status = document.querySelector('.search-status');
  let category = 'all';
  let index = [];

  fetch('search-index.json')
    .then(response => response.ok ? response.json() : [])
    .then(data => { index = data; apply(); })
    .catch(() => { index = []; });

  function normalized(value) {
    return value.toLocaleLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  }

  function searchable(row) {
    const link = row.querySelector('a');
    const item = index.find(entry => entry.url === link?.getAttribute('href'));
    return normalized([
      row.dataset.title || '', item?.title || '', item?.summary || '',
      ...(item?.aliases || [])
    ].join(' '));
  }

  function closeEnough(haystack, query) {
    if (!query || haystack.includes(query)) return true;
    const queryWords = query.split(/\s+/).filter(Boolean);
    const words = haystack.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    return queryWords.every(needle => words.some(word => {
      if (Math.abs(word.length - needle.length) > 2) return false;
      const previous = Array(needle.length + 1).fill(0).map((_, i) => i);
      for (let i = 1; i <= word.length; i += 1) {
        let diagonal = previous[0];
        previous[0] = i;
        for (let j = 1; j <= needle.length; j += 1) {
          const above = previous[j];
          previous[j] = Math.min(
            previous[j] + 1,
            previous[j - 1] + 1,
            diagonal + (word[i - 1] === needle[j - 1] ? 0 : 1)
          );
          diagonal = above;
        }
      }
      const allowance = needle.length >= 8 ? 2 : 1;
      return previous[needle.length] <= allowance;
    }));
  }

  function apply() {
    const query = normalized(input.value.trim());
    let shown = 0;
    rows.forEach(row => {
      const categoryMatch = category === 'all' || row.dataset.category === category;
      const textMatch = closeEnough(searchable(row), query);
      row.hidden = !(categoryMatch && textMatch);
      if (!row.hidden) shown += 1;
    });
    groups.forEach(group => {
      group.hidden = ![...group.querySelectorAll('li')].some(row => !row.hidden);
    });
    status.textContent = `${shown} article${shown === 1 ? '' : 's'} found`;
  }

  input.addEventListener('input', apply);
  buttons.forEach(button => button.addEventListener('click', () => {
    category = button.dataset.filter;
    buttons.forEach(candidate => candidate.classList.toggle('on', candidate === button));
    apply();
  }));
})();
