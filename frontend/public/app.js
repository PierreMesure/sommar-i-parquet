const elements = {
  search: document.querySelector("#search"),
  programType: document.querySelector("#program-type"),
  yearFrom: document.querySelector("#year-from"),
  yearTo: document.querySelector("#year-to"),
  yearFromOutput: document.querySelector("#year-from-output"),
  yearToOutput: document.querySelector("#year-to-output"),
  sort: document.querySelector("#sort"),
  returning: document.querySelector("#returning"),
  reset: document.querySelector("#reset"),
  count: document.querySelector("#result-count"),
  list: document.querySelector("#episode-list"),
  empty: document.querySelector("#empty-state"),
  template: document.querySelector("#episode-template"),
};

const EPISODE_URL_BASE = "https://www.sverigesradio.se/avsnitt/";
const IMAGE_URL_BASE = "https://static-cdn.sr.se";
const EPISODES_URL = document.querySelector('meta[name="episodes-url"]').content;

function formatDate(date) {
  return new Intl.DateTimeFormat("sv-SE", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(new Date(`${date}T12:00:00`));
}

function formatDuration(minutes) {
  return `${minutes} min`;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

let yearBounds;

function populateYearRange(episodes) {
  const years = episodes.map((episode) => Number(episode.date.slice(0, 4))).filter(Number.isFinite);
  yearBounds = { min: Math.min(...years), max: Math.max(...years) };
  for (const input of [elements.yearFrom, elements.yearTo]) {
    input.min = yearBounds.min;
    input.max = yearBounds.max;
  }
  elements.yearFrom.value = yearBounds.min;
  elements.yearTo.value = yearBounds.max;
  updateYearRangeLabels();
}

function updateYearRangeLabels() {
  elements.yearFromOutput.value = elements.yearFrom.value;
  elements.yearToOutput.value = elements.yearTo.value;
  const span = yearBounds.max - yearBounds.min;
  const fromPercent = ((Number(elements.yearFrom.value) - yearBounds.min) / span) * 100;
  const toPercent = ((Number(elements.yearTo.value) - yearBounds.min) / span) * 100;
  const range = elements.yearFrom.closest(".year-range");
  range.style.setProperty("--range-from", `${fromPercent}%`);
  range.style.setProperty("--range-to", `${toPercent}%`);
}

function displaySpeakers(episode) {
  return episode.speakers.join(", ");
}

function episodeUrl(episode) {
  return `${EPISODE_URL_BASE}${episode.id}`;
}

function imageUrl(episode) {
  return episode.image ? `${IMAGE_URL_BASE}${episode.image}` : null;
}

let records = [];
let virtualItems = [];
let itemOffsets = [];
let virtualHeight = 0;
let renderedStart;
let renderedEnd;
let renderTimer;
let scrollFrame;

function rowHeight() {
  return window.matchMedia("(max-width: 520px)").matches ? 186 : 180;
}

function itemHeight(item) {
  return item.type === "header" ? 54 : rowHeight();
}

function buildVirtualLayout() {
  itemOffsets = [];
  virtualHeight = 0;
  for (const item of virtualItems) {
    itemOffsets.push(virtualHeight);
    virtualHeight += itemHeight(item);
  }
  renderedStart = undefined;
  renderedEnd = undefined;
}

function firstVisibleItem(offset) {
  let low = 0;
  let high = itemOffsets.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (itemOffsets[middle] < offset) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

function render() {
  const episodes = records.map((record) => record.episode);
  const query = elements.search.value.trim().toLocaleLowerCase("sv-SE");
  const programType = elements.programType.value;
  const fromYear = Number(elements.yearFrom.value);
  const toYear = Number(elements.yearTo.value);
  const sort = elements.sort.value;
  const returningOnly = elements.returning.checked;

  const filtered = records.filter((record) => {
    const { episode } = record;
    return (
      (!query || record.searchable.includes(query)) &&
      (!programType || episode.type === programType) &&
      (Number(episode.date.slice(0, 4)) >= fromYear && Number(episode.date.slice(0, 4)) <= toYear) &&
      (!returningOnly || episode.returning)
    );
  });

  filtered.sort((left, right) => {
    if (returningOnly) {
      const groupComparison = left.group.localeCompare(right.group, "sv");
      if (groupComparison) return groupComparison;
      return sort === "oldest"
        ? left.episode.date.localeCompare(right.episode.date)
        : right.episode.date.localeCompare(left.episode.date);
    }
    if (sort === "speaker") return left.speakerName.localeCompare(right.speakerName, "sv");
    return sort === "oldest"
      ? left.episode.date.localeCompare(right.episode.date)
      : right.episode.date.localeCompare(left.episode.date);
  });

  elements.count.textContent = `${filtered.length.toLocaleString("sv-SE")} av ${episodes.length.toLocaleString("sv-SE")} program`;
  elements.empty.hidden = filtered.length > 0;
  const yearIsUnfiltered = fromYear === yearBounds.min && toYear === yearBounds.max;
  elements.reset.hidden = !query && !programType && yearIsUnfiltered && sort === "newest" && !returningOnly;
  virtualItems = [];
  let previousGroup;
  for (const record of filtered) {
    const group = returningOnly
      ? record.group
      : sort !== "speaker"
      ? record.episode.date.slice(0, 4)
        : null;
    if (group && group !== previousGroup) {
      virtualItems.push({ type: "header", label: group });
      previousGroup = group;
    }
    virtualItems.push({ type: "episode", record });
  }
  buildVirtualLayout();
  window.requestAnimationFrame(renderVisibleCards);
}

function scheduleRender() {
  window.clearTimeout(renderTimer);
  renderTimer = window.setTimeout(render, 120);
}

function renderVisibleCards() {
  const total = virtualItems.length;
  if (!total) {
    elements.list.replaceChildren();
    return;
  }

  const listTop = elements.list.getBoundingClientRect().top + window.scrollY;
  const viewportTop = Math.max(0, window.scrollY - listTop);
  const viewportBottom = viewportTop + window.innerHeight;
  const overscan = 8;
  const visibleStart = firstVisibleItem(viewportTop);
  const visibleEnd = Math.min(total, firstVisibleItem(viewportBottom) + 1);
  if (
    renderedStart !== undefined &&
    visibleStart >= renderedStart &&
    visibleEnd <= renderedEnd
  ) {
    return;
  }

  const start = Math.max(0, visibleStart - overscan);
  let end = Math.min(total, visibleEnd + overscan);
  end = Math.max(start + 1, end);
  const fragment = document.createDocumentFragment();
  const topSpacer = document.createElement("div");
  topSpacer.className = "virtual-spacer";
  topSpacer.style.height = `${itemOffsets[start]}px`;
  fragment.append(topSpacer);

  for (const item of virtualItems.slice(start, end)) {
    fragment.append(item.type === "header" ? createListHeader(item.label) : createEpisodeCard(item.record.episode));
  }

  const bottomSpacer = document.createElement("div");
  bottomSpacer.className = "virtual-spacer";
  bottomSpacer.style.height = `${Math.max(0, virtualHeight - (itemOffsets[end] ?? virtualHeight))}px`;
  fragment.append(bottomSpacer);
  elements.list.replaceChildren(fragment);
  renderedStart = start;
  renderedEnd = end;
}

function scheduleVisibleRender() {
  if (scrollFrame) return;
  scrollFrame = window.requestAnimationFrame(() => {
    scrollFrame = undefined;
    renderVisibleCards();
  });
}

function createListHeader(label) {
  const separator = document.createElement("h2");
  separator.className = "year-separator";
  separator.textContent = label;
  return separator;
}

function createEpisodeCard(episode) {
  const card = elements.template.content.cloneNode(true);
  const links = card.querySelectorAll('[data-role="episode-link"]');
  for (const link of links) link.href = episodeUrl(episode);

  const image = card.querySelector('[data-role="image"]');
  const fallback = card.querySelector(".image-fallback");
  fallback.textContent = episode.initials;
  // At most a few dozen cards are mounted by the virtual list, so eager image
  // loading avoids a visible delay when a card enters the viewport.
  image.loading = "eager";
  image.decoding = "async";
  if (episode.image) {
    image.src = imageUrl(episode);
    image.alt = `Bild till ${displaySpeakers(episode)}`;
    image.addEventListener("error", () => {
      image.hidden = true;
      fallback.hidden = false;
    });
  } else {
    image.hidden = true;
    fallback.hidden = false;
  }

  const programType = card.querySelector('[data-role="program-type"]');
  const date = card.querySelector('[data-role="date"]');
  const speakers = card.querySelector('[data-role="speakers"]');
  programType.textContent = episode.type || "P1";
  date.textContent = formatDate(episode.date);
  card.querySelector('[data-role="duration"]').textContent = formatDuration(episode.minutes);
  if (elements.returning.checked) {
    card.firstElementChild.classList.add("returning-episode");
    programType.hidden = true;
    date.hidden = true;
    speakers.textContent = `${episode.type || "P1"} · ${formatDate(episode.date)}`;
  } else {
    speakers.textContent = displaySpeakers(episode);
  }
  card.querySelector('[data-role="summary"]').textContent = episode.description;
  return card.firstElementChild;
}

async function loadEpisodes() {
  const response = await fetch(EPISODES_URL);
  if (!response.ok) throw new Error("Kunde inte läsa programdata.");
  const payload = await response.json();
  return payload.episodes;
}

for (const input of [elements.search, elements.programType, elements.sort, elements.returning]) {
  input.addEventListener("input", scheduleRender);
  input.addEventListener("change", render);
}
for (const [input, other, direction] of [
  [elements.yearFrom, elements.yearTo, "from"],
  [elements.yearTo, elements.yearFrom, "to"],
]) {
  input.addEventListener("input", () => {
    if (direction === "from" && Number(input.value) > Number(other.value)) other.value = input.value;
    if (direction === "to" && Number(input.value) < Number(other.value)) other.value = input.value;
    updateYearRangeLabels();
    scheduleRender();
  });
  input.addEventListener("change", render);
}
window.addEventListener("scroll", scheduleVisibleRender, { passive: true });
window.addEventListener("resize", () => {
  buildVirtualLayout();
  scheduleVisibleRender();
});
elements.reset.addEventListener("click", () => {
  elements.search.value = "";
  elements.programType.value = "";
  elements.yearFrom.value = yearBounds.min;
  elements.yearTo.value = yearBounds.max;
  updateYearRangeLabels();
  elements.sort.value = "newest";
  elements.returning.checked = false;
  render();
});

try {
  const episodes = await loadEpisodes();
  populateYearRange(episodes);
  records = episodes.map((episode) => {
    const speakerName = displaySpeakers(episode);
    return {
      episode,
      speakerName,
      group: episode.group || "",
      searchable: [speakerName, episode.description]
        .join(" ")
        .toLocaleLowerCase("sv-SE"),
    };
  });
  render();
} catch (error) {
  elements.count.textContent = error.message;
}
