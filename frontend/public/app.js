const elements = {
  search: document.querySelector("#search"),
  programType: document.querySelector("#program-type"),
  yearFrom: document.querySelector("#year-from"),
  yearTo: document.querySelector("#year-to"),
  yearFromOutput: document.querySelector("#year-from-output"),
  yearToOutput: document.querySelector("#year-to-output"),
  sort: document.querySelector("#sort"),
  returning: document.querySelector("#returning"),
  gender: document.querySelector("#gender"),
  citizenship: document.querySelector("#citizenship"),
  occupation: document.querySelector("#occupation"),
  ageFrom: document.querySelector("#age-from"),
  ageTo: document.querySelector("#age-to"),
  ageFromOutput: document.querySelector("#age-from-output"),
  ageToOutput: document.querySelector("#age-to-output"),
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
let ageBounds;
let speakersById = {};

function populateYearRange(episodes) {
  const years = episodes.map((episode) => Number(episode.date.slice(0, 4))).filter(Number.isFinite);
  yearBounds = years.length
    ? { min: Math.min(...years), max: Math.max(...years) }
    : { min: 0, max: 0 };
  for (const input of [elements.yearFrom, elements.yearTo]) {
    input.min = yearBounds.min;
    input.max = yearBounds.max;
  }
  elements.yearFrom.value = yearBounds.min;
  elements.yearTo.value = yearBounds.max;
  updateYearRangeLabels();
}

function populateAgeRange(episodes) {
  const ages = episodes.flatMap((episode) => episode.ages || []).filter(Number.isFinite);
  ageBounds = ages.length
    ? { min: Math.min(...ages), max: Math.max(...ages) }
    : { min: 0, max: 0 };
  for (const input of [elements.ageFrom, elements.ageTo]) {
    input.min = ageBounds.min;
    input.max = ageBounds.max;
    input.disabled = !ages.length;
  }
  elements.ageFrom.value = ageBounds.min;
  elements.ageTo.value = ageBounds.max;
  updateAgeRangeLabels();
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

function updateAgeRangeLabels() {
  elements.ageFromOutput.value = ageBounds?.min ? `${elements.ageFrom.value} år` : "–";
  elements.ageToOutput.value = ageBounds?.max ? `${elements.ageTo.value} år` : "–";
  const span = (ageBounds?.max || 0) - (ageBounds?.min || 0);
  const fromPercent = span ? ((Number(elements.ageFrom.value) - ageBounds.min) / span) * 100 : 0;
  const toPercent = span ? ((Number(elements.ageTo.value) - ageBounds.min) / span) * 100 : 100;
  const range = elements.ageFrom.closest(".age-range");
  range.style.setProperty("--range-from", `${fromPercent}%`);
  range.style.setProperty("--range-to", `${toPercent}%`);
}

function displaySpeakers(episode) {
  return (episode.speakers || [])
    .map((qid) => speakersById[qid]?.name || qid)
    .join(", ");
}

function populateSelect(select, values) {
  const options = unique(values).sort((left, right) => left.localeCompare(right, "sv"));
  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
}

function populateSpeakerFilters() {
  const speakers = Object.values(speakersById);
  populateSelect(elements.gender, speakers.map((speaker) => speaker.gender));
  populateSelect(
    elements.citizenship,
    speakers.flatMap((speaker) => speaker.citizenships || []),
  );
  populateSelect(
    elements.occupation,
    speakers.flatMap((speaker) => speaker.occupations || []),
  );
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
  const gender = elements.gender.value;
  const citizenship = elements.citizenship.value;
  const occupation = elements.occupation.value;
  const fromAge = Number(elements.ageFrom.value);
  const toAge = Number(elements.ageTo.value);
  const ageIsUnfiltered = !ageBounds || (fromAge === ageBounds.min && toAge === ageBounds.max);

  const filtered = records.filter((record) => {
    const { episode } = record;
    return (
      (!query || record.searchable.includes(query)) &&
      (!programType || episode.type === programType) &&
      (Number(episode.date.slice(0, 4)) >= fromYear && Number(episode.date.slice(0, 4)) <= toYear) &&
      (!returningOnly || record.returning) &&
      (!gender || record.genders.has(gender)) &&
      (!citizenship || record.citizenships.has(citizenship)) &&
      (!occupation || record.occupations.has(occupation)) &&
      (ageIsUnfiltered || (episode.ages || []).some((age) => age >= fromAge && age <= toAge))
    );
  });

  filtered.sort((left, right) => {
    if (returningOnly) {
      const groupComparison = left.group.name.localeCompare(right.group.name, "sv");
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
  elements.reset.hidden =
    !query &&
    !programType &&
    yearIsUnfiltered &&
    sort === "newest" &&
    !returningOnly &&
    !gender &&
    !citizenship &&
    !occupation &&
    ageIsUnfiltered;
  virtualItems = [];
  let previousGroup;
  for (const record of filtered) {
    const group = returningOnly
      ? record.group
      : sort !== "speaker"
      ? record.episode.date.slice(0, 4)
        : null;
    const groupKey = group && typeof group === "object" ? group.id : group;
    const groupLabel = group && typeof group === "object" ? group.name : group;
    if (groupKey && groupKey !== previousGroup) {
      virtualItems.push({ type: "header", label: groupLabel });
      previousGroup = groupKey;
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
  const wikiLinks = card.querySelector('[data-role="speaker-wiki-links"]');
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
  for (const qid of episode.speakers || []) {
    const speaker = speakersById[qid];
    if (!speaker?.wiki) continue;
    const link = document.createElement("a");
    link.className = "speaker-wiki-link";
    link.href = speaker.wiki;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "Wikipedia";
    link.setAttribute("aria-label", `Wikipedia: ${speaker.name}`);
    wikiLinks.append(link);
  }
  card.querySelector('[data-role="summary"]').textContent = episode.description;
  return card.firstElementChild;
}

async function loadArchive() {
  const response = await fetch(EPISODES_URL);
  if (!response.ok) throw new Error("Kunde inte läsa programdata.");
  return response.json();
}

for (const input of [
  elements.search,
  elements.programType,
  elements.sort,
  elements.returning,
  elements.gender,
  elements.citizenship,
  elements.occupation,
]) {
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
for (const [input, other, direction] of [
  [elements.ageFrom, elements.ageTo, "from"],
  [elements.ageTo, elements.ageFrom, "to"],
]) {
  input.addEventListener("input", () => {
    if (direction === "from" && Number(input.value) > Number(other.value)) other.value = input.value;
    if (direction === "to" && Number(input.value) < Number(other.value)) other.value = input.value;
    updateAgeRangeLabels();
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
  elements.gender.value = "";
  elements.citizenship.value = "";
  elements.occupation.value = "";
  elements.ageFrom.value = ageBounds.min;
  elements.ageTo.value = ageBounds.max;
  updateAgeRangeLabels();
  render();
});

try {
  const archive = await loadArchive();
  const episodes = Array.isArray(archive?.episodes) ? archive.episodes : [];
  speakersById = archive?.speakers && typeof archive.speakers === "object"
    ? archive.speakers
    : {};
  populateYearRange(episodes);
  populateAgeRange(episodes);
  populateSpeakerFilters();
  records = episodes.map((episode) => {
    const speakerRecords = (episode.speakers || [])
      .map((qid) => ({ qid, ...speakersById[qid] }))
      .filter((speaker) => speaker.name);
    const returningSpeakers = speakerRecords.filter((speaker) => speaker.count > 1);
    const speakerName = displaySpeakers(episode);
    return {
      episode,
      speakerName,
      returning: returningSpeakers.length > 0,
      group: returningSpeakers.length
        ? { id: returningSpeakers[0].qid, name: returningSpeakers[0].name }
        : null,
      genders: new Set(speakerRecords.map((speaker) => speaker.gender).filter(Boolean)),
      citizenships: new Set(
        speakerRecords.flatMap((speaker) => speaker.citizenships || []),
      ),
      occupations: new Set(
        speakerRecords.flatMap((speaker) => speaker.occupations || []),
      ),
      searchable: [episode.description || ""]
        .join(" ")
        .toLocaleLowerCase("sv-SE"),
    };
  });
  render();
} catch (error) {
  elements.count.textContent = error.message;
}
