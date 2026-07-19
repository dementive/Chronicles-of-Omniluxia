(() => {
  const getJson = url => fetch(url).then(response => response.ok ? response.json() : []);
  const articleIndex = getJson('search-index.json').catch(() => []);
  const timelineIndex = getJson('timeline-events.json').catch(() => []);

  function normalized(value) {
    return value.toLocaleLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  }

  // Random discovery works from every page; ordinary hrefs remain as no-JS fallbacks.
  document.querySelectorAll('[data-random-page]').forEach(link => {
    link.addEventListener('click', async event => {
      const index = await articleIndex;
      const scope = link.dataset.randomScope || 'all';
      const choices = scope === 'all' ? index : index.filter(item => item.category === scope);
      if (!choices.length) return;
      event.preventDefault();
      window.location.href = choices[Math.floor(Math.random() * choices.length)].url;
    });
  });

  document.querySelectorAll('[data-random-moment]').forEach(link => {
    link.addEventListener('click', async event => {
      const moments = await timelineIndex;
      if (!moments.length) return;
      event.preventDefault();
      const pick = Math.floor(Math.random() * moments.length);
      window.location.href = `timeline.html?random=${pick}#chronology-explorer`;
    });
  });

  const dayCard = document.querySelector('[data-on-this-day]');
  if (dayCard) {
    const date = dayCard.querySelector('[data-day-date]');
    const event = dayCard.querySelector('[data-day-event]');
    const reroll = dayCard.querySelector('[data-day-reroll]');
    let moments = [];
    let current = -1;

    function showMoment(index) {
      if (!moments.length) return;
      current = ((index % moments.length) + moments.length) % moments.length;
      date.textContent = moments[current].date;
      event.innerHTML = moments[current].html;
    }

    timelineIndex.then(data => {
      moments = data;
      const now = new Date();
      const dayNumber = Math.floor(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 86400000);
      showMoment(dayNumber);
    });

    reroll?.addEventListener('click', () => {
      if (moments.length < 2) return;
      let next = current;
      while (next === current) next = Math.floor(Math.random() * moments.length);
      showMoment(next);
    });
  }

  const explorer = document.querySelector('[data-timeline-explorer]');
  if (explorer) {
    const input = explorer.querySelector('#timeline-search');
    const results = explorer.querySelector('[data-timeline-results]');
    const status = explorer.querySelector('.timeline-status');
    const eraButtons = [...explorer.querySelectorAll('[data-era]')];
    const randomButton = explorer.querySelector('[data-random-event]');
    const moreButton = explorer.querySelector('[data-timeline-more]');
    let moments = [];
    let era = 'all';
    let limit = 20;

    function matchesEra(item) {
      if (era === 'all') return true;
      if (era === 'blc') return /\bBLC\b/i.test(item.date);
      if (era === 'lc') return /\bLC\b/i.test(item.date) && !/\bBLC\b/i.test(item.date);
      return item.date.includes('~');
    }

    function card(item) {
      return `<article class="timeline-card"><time>${item.date}</time><div>${item.html}</div></article>`;
    }

    function filtered() {
      const query = normalized(input.value.trim());
      return moments.filter(item => matchesEra(item) && (!query || normalized(`${item.date} ${item.text}`).includes(query)));
    }

    function render() {
      const visible = filtered();
      results.innerHTML = visible.slice(0, limit).map(card).join('');
      status.textContent = `${visible.length} historical moment${visible.length === 1 ? '' : 's'} found`;
      moreButton.hidden = visible.length <= limit;
    }

    timelineIndex.then(data => {
      moments = data;
      const params = new URLSearchParams(window.location.search);
      const random = Number(params.get('random'));
      if (params.has('random') && Number.isInteger(random) && random >= 0 && random < moments.length) {
        results.innerHTML = card(moments[random]);
        status.textContent = 'Random historical moment';
        moreButton.hidden = true;
      } else {
        render();
      }
    });

    input.addEventListener('input', () => { limit = 20; render(); });
    eraButtons.forEach(button => button.addEventListener('click', () => {
      era = button.dataset.era;
      limit = 20;
      eraButtons.forEach(candidate => candidate.classList.toggle('on', candidate === button));
      render();
    }));
    randomButton.addEventListener('click', () => {
      const choices = filtered();
      if (!choices.length) return;
      results.innerHTML = card(choices[Math.floor(Math.random() * choices.length)]);
      status.textContent = 'Random historical moment';
      moreButton.hidden = true;
    });
    moreButton.addEventListener('click', () => { limit += 20; render(); });
  }

  const input = document.querySelector('#wiki-search');
  if (!input) return;

  const rows = [...document.querySelectorAll('.idx-list li')];
  const groups = [...document.querySelectorAll('[data-category-group]')];
  const buttons = [...document.querySelectorAll('[data-filter]')];
  const status = document.querySelector('.search-status');
  let category = 'all';
  let index = [];

  articleIndex.then(data => { index = data; apply(); });

  function searchable(row) {
    const link = row.querySelector('a');
    const item = index.find(entry => entry.url === link?.getAttribute('href'));
    return normalized([row.dataset.title || '', item?.title || '', item?.summary || '', ...(item?.aliases || [])].join(' '));
  }

  function closeEnough(haystack, query) {
    if (!query || haystack.includes(query)) return true;
    const queryWords = query.split(/\s+/).filter(Boolean);
    const words = haystack.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    return queryWords.every(needle => words.some(word => {
      if (Math.abs(word.length - needle.length) > 2) return false;
      const previous = Array(needle.length + 1).fill(0).map((_, i) => i);
      for (let i = 1; i <= word.length; i += 1) {
        let diagonal = previous[0]; previous[0] = i;
        for (let j = 1; j <= needle.length; j += 1) {
          const above = previous[j];
          previous[j] = Math.min(previous[j] + 1, previous[j - 1] + 1, diagonal + (word[i - 1] === needle[j - 1] ? 0 : 1));
          diagonal = above;
        }
      }
      return previous[needle.length] <= (needle.length >= 8 ? 2 : 1);
    }));
  }

  function apply() {
    const query = normalized(input.value.trim());
    let shown = 0;
    rows.forEach(row => {
      const visible = (category === 'all' || row.dataset.category === category) && closeEnough(searchable(row), query);
      row.hidden = !visible;
      if (visible) shown += 1;
    });
    groups.forEach(group => { group.hidden = ![...group.querySelectorAll('li')].some(row => !row.hidden); });
    status.textContent = `${shown} article${shown === 1 ? '' : 's'} found`;
  }

  input.addEventListener('input', apply);
  buttons.forEach(button => button.addEventListener('click', () => {
    category = button.dataset.filter;
    buttons.forEach(candidate => candidate.classList.toggle('on', candidate === button));
    apply();
  }));
})();
