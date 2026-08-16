const elements = {
  search: document.querySelector("#search"),
  programType: document.querySelector("#program-type"),
  listenersHost: document.querySelector("#listeners-host"),
  yearFrom: document.querySelector("#year-from"),
  yearTo: document.querySelector("#year-to"),
  yearFromOutput: document.querySelector("#year-from-output"),
  yearToOutput: document.querySelector("#year-to-output"),
  sort: document.querySelector("#sort"),
  returning: document.querySelector("#returning"),
  gender: document.querySelector("#gender"),
  citizenship: document.querySelector("#citizenship"),
  occupation: document.querySelector("#occupation"),
  theme: document.querySelector("#theme"),
  topic: document.querySelector("#topic"),
  themeLabel: document.querySelector('[data-role="theme-label"]'),
  topicLabel: document.querySelector('[data-role="topic-label"]'),
  filters: document.querySelector("#filters"),
  filtersToggle: document.querySelector("#filters-toggle"),
  contentFilters: document.querySelector("#content-filters"),
  mapToggle: document.querySelector("#map-toggle"),
  mapLabel: document.querySelector('[data-role="map-label"]'),
  mapDrawer: document.querySelector("#map-drawer"),
  ageFrom: document.querySelector("#age-from"),
  ageTo: document.querySelector("#age-to"),
  ageFromOutput: document.querySelector("#age-from-output"),
  ageToOutput: document.querySelector("#age-to-output"),
  reset: document.querySelector("#reset"),
  count: document.querySelector("#result-count"),
  list: document.querySelector("#episode-list"),
  empty: document.querySelector("#empty-state"),
  template: document.querySelector("#episode-template"),
  archiveView: document.querySelector("#archive-view"),
  map: document.querySelector("#episode-map"),
};

function setFiltersOpen(open) {
  elements.filters.classList.toggle("is-hidden", !open);
  elements.filtersToggle.setAttribute("aria-expanded", String(open));
  elements.filtersToggle.textContent = open ? "Dölj filter" : "Visa filter";
}

setFiltersOpen(!window.matchMedia("(max-width: 520px)").matches);

function updateTopicSortAvailability() {
  const option = elements.sort.querySelector('option[value="topic-match"]');
  if (!option) return;
  const enabled = selectedValues(elements.topic).length > 0 || selectedValues(elements.theme).length > 0;
  option.disabled = !enabled;
  if (!enabled && elements.sort.value === "topic-match") elements.sort.value = "newest";
}

const EPISODE_URL_BASE = "https://www.sverigesradio.se/avsnitt/";
const IMAGE_URL_BASE = "https://static-cdn.sr.se";
const EPISODES_URL = document.querySelector('meta[name="episodes-url"]').content;
const TOPICS_URL = document.querySelector('meta[name="topics-url"]').content;
const TOPIC_COLOURS = [
  "#a1471d", "#2f6a4d", "#b38324", "#486b8a", "#884f68", "#53786e",
  "#b75b43", "#6e6b35", "#66548b", "#34747d", "#95622f", "#4e7350",
];

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

function seasonLabel(episode) {
  const calendarYear = Number(String(episode.date || "").slice(0, 4));
  let year = Number(episode.year);
  if (!Number.isFinite(year)) {
    year = episode.type === "Vinter" && String(episode.date || "").slice(5, 7) === "01"
      ? calendarYear - 1
      : calendarYear;
  }
  return episode.type === "Vinter" ? `VT${year}` : String(year);
}

function capitalise(value) {
  const text = String(value || "");
  return text ? `${text[0].toLocaleUpperCase("sv-SE")}${text.slice(1)}` : text;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

let yearBounds;
let ageBounds;
let speakersById = {};
let topicsById = {};
let analysisByEpisode = {};
let episodesById = new Map();
let selectedEpisodeId;
let mapOnlyEpisodeId;
let filteredRecords = [];
let mapPoints = [];
let mapFrame;

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
    option.textContent = select === elements.occupation || select === elements.gender ? capitalise(value) : value;
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

function topicLabel(topicId) {
  return topicsById[String(topicId)]?.label || `Ämne ${topicId}`;
}

function topicColour(topicId) {
  if (topicId == null || topicId === "") return "#b8b1a7";
  const text = String(topicId);
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
  }
  return TOPIC_COLOURS[Math.abs(hash) % TOPIC_COLOURS.length];
}

function selectedValues(select) {
  return [...select.selectedOptions].map((option) => option.value);
}

function setSelectedValues(select, values) {
  const selected = new Set(values.map(String));
  for (const option of select.options) option.selected = selected.has(option.value);
}

function toggleSelectedValue(select, value) {
  const option = [...select.options].find((candidate) => candidate.value === String(value));
  if (option) option.selected = !option.selected;
}

function populateTopicFilters() {
  const topics = Object.entries(topicsById)
    .sort(([, left], [, right]) => left.label.localeCompare(right.label, "sv"));
  const themes = unique(topics.map(([, topic]) => topic.parent))
    .sort((left, right) => left.localeCompare(right, "sv"));
  elements.theme.replaceChildren();
  elements.topic.replaceChildren();
  const themeEpisodeIds = new Map(themes.map((theme) => [theme, new Set()]));
  for (const [episodeId, analysis] of Object.entries(analysisByEpisode)) {
    const episodeThemes = new Set(
      (analysis?.topics || [])
        .map(([topicId]) => topicsById[String(topicId)]?.parent)
        .filter(Boolean),
    );
    for (const theme of episodeThemes) themeEpisodeIds.get(theme)?.add(episodeId);
  }
  for (const theme of themes) {
    const option = document.createElement("option");
    option.value = theme;
    const count = themeEpisodeIds.get(theme)?.size || 0;
    option.textContent = theme;
    option.title = `${count} program`;
    elements.theme.append(option);
  }
  refreshLeafTopicOptions();
  if (topics.length) {
    elements.contentFilters.disabled = false;
    elements.contentFilters.classList.remove("filter-group-disabled");
  }
  if (elements.themeLabel) elements.themeLabel.textContent = `Övergripande tema (${themes.length} st.)`;
  if (elements.topicLabel) elements.topicLabel.textContent = `Specifikt ämne (${topics.length} st.)`;
  renderTopicList();
}

function refreshLeafTopicOptions() {
  const selectedTopics = selectedValues(elements.topic);
  const selectedThemes = new Set(selectedValues(elements.theme));
  const topics = Object.entries(topicsById)
    .filter(([, topic]) => !selectedThemes.size || selectedThemes.has(topic.parent))
    .sort(([, left], [, right]) => left.label.localeCompare(right.label, "sv"));
  elements.topic.replaceChildren();
  for (const [topicId, topic] of topics) {
    const option = document.createElement("option");
    option.value = topicId;
    option.textContent = topic.label;
    option.title = `${topic.episodes || 0} program`;
    option.style.color = topicColour(topicId);
    option.selected = selectedTopics.includes(topicId);
    elements.topic.append(option);
  }
}

function episodeUrl(episode) {
  return `${EPISODE_URL_BASE}${episode.id}`;
}

function imageUrl(episode) {
  return episode.image ? `${IMAGE_URL_BASE}${episode.image}` : null;
}

let records = [];
const PAGE_SIZE = 100;
let recordLimit = PAGE_SIZE;
let filteredRecordCount = 0;
let loadedRecordCount = 0;
let lastFilterSignature;
let loadMorePending = false;
let virtualItems = [];
let itemOffsets = [];
let virtualHeight = 0;
let renderedStart;
let renderedEnd;
let expandedDetailHeight = 0;
let renderTimer;
let scrollFrame;

function rowHeight() {
  return window.matchMedia("(max-width: 520px)").matches ? 186 : 180;
}

function itemHeight(item) {
  if (item.type === "header") return 54;
  return item.record?.episode?.id === selectedEpisodeId ? rowHeight() + expandedDetailHeight : rowHeight();
}

function episodeAge(record) {
  const ages = (record.episode.ages || [])
    .map(Number)
    .filter((age) => Number.isFinite(age));
  return ages.length ? ages.reduce((sum, age) => sum + age, 0) / ages.length : null;
}

function topicMatch(record, selectedTopics, selectedThemes) {
  const rows = analysisByEpisode[String(record.episode.id)]?.topics || [];
  const candidates = rows.filter(([topicId]) => {
    if (selectedTopics.length) return selectedTopics.includes(String(topicId));
    if (selectedThemes.length) return selectedThemes.includes(topicsById[String(topicId)]?.parent);
    return false;
  });
  return candidates.reduce((best, [, share]) => Math.max(best, Number(share) || 0), 0);
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
  const listenersHostOnly = Boolean(elements.listenersHost?.checked);
  const fromYear = Number(elements.yearFrom.value);
  const toYear = Number(elements.yearTo.value);
  const returningOnly = elements.returning.checked;
  const gender = elements.gender.value;
  const citizenship = elements.citizenship.value;
  const occupation = elements.occupation.value;
  const selectedTopics = selectedValues(elements.topic);
  const selectedThemes = selectedValues(elements.theme);
  updateTopicSortAvailability();
  const sort = elements.sort.value;
  const fromAge = Number(elements.ageFrom.value);
  const toAge = Number(elements.ageTo.value);
  const ageIsUnfiltered = !ageBounds || (fromAge === ageBounds.min && toAge === ageBounds.max);
  const filterSignature = JSON.stringify([
    query,
    programType,
    listenersHostOnly,
    fromYear,
    toYear,
    sort,
    returningOnly,
    gender,
    citizenship,
    occupation,
    selectedTopics,
    selectedThemes,
    mapOnlyEpisodeId,
    fromAge,
    toAge,
  ]);
  if (filterSignature !== lastFilterSignature) {
    recordLimit = PAGE_SIZE;
    lastFilterSignature = filterSignature;
  }

  const filtered = records.filter((record) => {
    const { episode } = record;
    return (
      (!query || record.searchable.includes(query)) &&
      (!programType || episode.type === programType) &&
      (!listenersHostOnly || Boolean(episode.is_listeners_host)) &&
      (Number(episode.date.slice(0, 4)) >= fromYear && Number(episode.date.slice(0, 4)) <= toYear) &&
      (!returningOnly || record.returning) &&
      (!gender || record.genders.has(gender)) &&
      (!citizenship || record.citizenships.has(citizenship)) &&
      (!occupation || record.occupations.has(occupation)) &&
      // A leaf selection is a refinement of the selected parent theme(s), so
      // it must take precedence rather than being ORed with the broad match.
      (!selectedTopics.length
        ? (!selectedThemes.length || selectedThemes.some((theme) => record.topicParents.has(theme)))
        : selectedTopics.some((topicId) => record.topicIds.has(topicId))) &&
      (!mapOnlyEpisodeId || Number(episode.id) === Number(mapOnlyEpisodeId)) &&
      (ageIsUnfiltered || (episode.ages || []).some((age) => age >= fromAge && age <= toAge))
    );
  });

  filtered.sort((left, right) => {
    const compareSort = () => {
      if (sort === "speaker") return left.speakerName.localeCompare(right.speakerName, "sv");
      if (sort === "age-youngest" || sort === "age-oldest") {
        const leftAge = episodeAge(left);
        const rightAge = episodeAge(right);
        if (leftAge == null && rightAge != null) return 1;
        if (leftAge != null && rightAge == null) return -1;
        if (leftAge != null && rightAge != null && leftAge !== rightAge) {
          return sort === "age-youngest" ? leftAge - rightAge : rightAge - leftAge;
        }
      }
      if (sort === "topic-match" && (selectedTopics.length || selectedThemes.length)) {
        const matchDifference = topicMatch(right, selectedTopics, selectedThemes) - topicMatch(left, selectedTopics, selectedThemes);
        if (matchDifference) return matchDifference;
      }
      return sort === "oldest"
        ? left.episode.date.localeCompare(right.episode.date)
        : right.episode.date.localeCompare(left.episode.date);
    };
    if (returningOnly) {
      const groupComparison = left.group.name.localeCompare(right.group.name, "sv");
      if (groupComparison) return groupComparison;
      return compareSort();
    }
    return compareSort();
  });

  filteredRecords = filtered;
  const visibleRecords = filtered.slice(0, recordLimit);
  filteredRecordCount = filtered.length;
  loadedRecordCount = visibleRecords.length;
  elements.count.textContent = filtered.length === episodes.length
    ? `${filtered.length.toLocaleString("sv-SE")} program`
    : `${filtered.length.toLocaleString("sv-SE")} av ${episodes.length.toLocaleString("sv-SE")} program`;
  elements.empty.hidden = filtered.length > 0;
  const yearIsUnfiltered = fromYear === yearBounds.min && toYear === yearBounds.max;
  elements.reset.hidden =
    !query &&
    !programType &&
    !listenersHostOnly &&
    yearIsUnfiltered &&
    sort === "newest" &&
    !returningOnly &&
    !gender &&
    !citizenship &&
    !occupation &&
    !selectedTopics.length &&
    !selectedThemes.length &&
    ageIsUnfiltered &&
    !mapOnlyEpisodeId;
  virtualItems = [];
  let previousGroup;
  for (const record of visibleRecords) {
    const group = returningOnly
      ? record.group
      : !["speaker", "age-youngest", "age-oldest", "topic-match"].includes(sort)
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
  scheduleMapRender();
}

function setFilterControlsLocked(locked) {
  document.querySelectorAll(".filters input, .filters select, .filters button").forEach((control) => {
    if (control === elements.mapToggle) return;
    control.disabled = locked;
  });
  elements.contentFilters.classList.toggle("filter-group-disabled", locked);
}

function selectEpisode(episodeId) {
  elements.search.value = "";
  elements.programType.value = "";
  elements.yearFrom.value = yearBounds.min;
  elements.yearTo.value = yearBounds.max;
  elements.sort.value = "newest";
  elements.returning.checked = false;
  elements.gender.value = "";
  elements.citizenship.value = "";
  elements.occupation.value = "";
  setSelectedValues(elements.theme, []);
  setSelectedValues(elements.topic, []);
  elements.ageFrom.value = ageBounds.min;
  elements.ageTo.value = ageBounds.max;
  updateYearRangeLabels();
  updateAgeRangeLabels();
  mapOnlyEpisodeId = Number(episodeId);
  selectedEpisodeId = Number(episodeId);
  expandedDetailHeight = 0;
  setFilterControlsLocked(true);
  render();
  scheduleMapRender();
}

function openEpisodeDetails(episodeId) {
  selectedEpisodeId = selectedEpisodeId === Number(episodeId) ? undefined : Number(episodeId);
  expandedDetailHeight = 0;
  render();
}

function filterByTopic(topicId) {
  elements.reset.click();
  // A topic bubble is an explicit leaf-topic selection. Clear any theme
  // selection first so the leaf option is available even when it belongs to
  // another theme.
  setSelectedValues(elements.theme, []);
  refreshLeafTopicOptions();
  setSelectedValues(elements.topic, [String(topicId)]);
  render();
  window.requestAnimationFrame(() => {
    [...elements.topic.options].find((option) => option.value === String(topicId))
      ?.scrollIntoView({ block: "nearest" });
  });
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
  let selectedSpacer;
  const topSpacer = document.createElement("div");
  topSpacer.className = "virtual-spacer";
  topSpacer.style.height = `${itemOffsets[start]}px`;
  fragment.append(topSpacer);

  for (const item of virtualItems.slice(start, end)) {
  if (item.type === "header") {
      fragment.append(createListHeader(item.label));
      continue;
    }
    const card = createEpisodeCard(item.record.episode);
    fragment.append(card);
    if (item.record.episode.id === selectedEpisodeId) {
      const detailSpacer = document.createElement("div");
      detailSpacer.className = "expanded-detail-spacer";
      fragment.append(detailSpacer);
      selectedSpacer = detailSpacer;
    }
  }

  const bottomSpacer = document.createElement("div");
  bottomSpacer.className = "virtual-spacer";
  bottomSpacer.style.height = `${Math.max(0, virtualHeight - (itemOffsets[end] ?? virtualHeight))}px`;
  fragment.append(bottomSpacer);
  elements.list.replaceChildren(fragment);
  let layoutChanged = false;
  if (selectedSpacer) {
    const selectedCard = elements.list.querySelector(`[data-episode-id="${selectedEpisodeId}"]`);
    const measuredHeight = selectedCard?.querySelector(".episode-detail")?.offsetHeight || 0;
    selectedSpacer.style.height = `${measuredHeight}px`;
    if (measuredHeight !== expandedDetailHeight) {
      expandedDetailHeight = measuredHeight;
      buildVirtualLayout();
      layoutChanged = true;
    }
  }
  renderedStart = layoutChanged ? undefined : start;
  renderedEnd = layoutChanged ? undefined : end;
  if (layoutChanged) window.requestAnimationFrame(renderVisibleCards);
}

function scheduleVisibleRender() {
  if (scrollFrame) return;
  scrollFrame = window.requestAnimationFrame(() => {
    scrollFrame = undefined;
    renderVisibleCards();
  });
}

function loadMoreNearPageEnd() {
  if (loadMorePending || loadedRecordCount >= filteredRecordCount) return;
  const remaining = document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
  if (remaining > Math.max(900, window.innerHeight)) return;
  loadMorePending = true;
  recordLimit += PAGE_SIZE;
  render();
  window.requestAnimationFrame(() => {
    loadMorePending = false;
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
  const article = card.firstElementChild;
  article.dataset.episodeId = episode.id;
  article.classList.toggle("is-expanded", Number(episode.id) === selectedEpisodeId);
  article.addEventListener("click", () => {
    openEpisodeDetails(episode.id);
  });
  for (const control of article.querySelectorAll('[data-role="episode-open"]')) {
    control.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openEpisodeDetails(episode.id);
      }
    });
  }

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
  const tags = card.querySelector('[data-role="topic-tags"]');
  const listen = card.querySelector('[data-role="listen-link"]');
  listen.href = episodeUrl(episode);
  const detailsToggle = card.querySelector('[data-role="details-toggle"]');
  detailsToggle.textContent = Number(episode.id) === selectedEpisodeId ? "Göm detaljer ↑" : "Detaljer ↓";
  detailsToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    openEpisodeDetails(episode.id);
  });
  const analysis = analysisByEpisode[String(episode.id)];
  const topicRows = analysis?.topics || [];
  for (const [topicId] of topicRows.slice(0, 2)) {
    const tag = document.createElement("button");
    tag.className = "topic-tag";
    tag.type = "button";
    tag.textContent = topicLabel(topicId);
    tag.style.background = topicColour(topicId);
    tag.style.borderColor = topicColour(topicId);
    tag.style.color = "#fffdf9";
    tag.addEventListener("click", (event) => {
      event.stopPropagation();
      filterByTopic(topicId);
    });
    tags.append(tag);
  }
  if (topicRows.length > 2) {
    const more = document.createElement("span");
    more.className = "topic-tag topic-more";
    more.textContent = `+${topicRows.length - 2}`;
    tags.append(more);
  }
  if (Number(episode.id) === selectedEpisodeId) article.append(renderEpisodeDetail(episode.id));
  return article;
}

function renderTopicList() {
  // The select boxes are the topic browser; keep this hook so old filter
  // events remain harmless after removing the separate exploration panel.
}

function scheduleMapRender() {
  if (elements.mapDrawer?.hidden || mapFrame) return;
  mapFrame = window.requestAnimationFrame(() => {
    mapFrame = undefined;
    renderMap();
  });
}

function renderMap() {
  const canvas = elements.map;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const allAnalyses = Object.values(analysisByEpisode);
  const xs = allAnalyses.map((row) => Number(row.x)).filter(Number.isFinite);
  const ys = allAnalyses.map((row) => Number(row.y)).filter(Number.isFinite);
  if (!xs.length || !ys.length) {
    mapPoints = [];
    return;
  }
  const padding = 24;
  const width = Math.max(1, bounds.width - padding * 2);
  const height = Math.max(1, bounds.height - padding * 2);
  const allRecords = records;
  const mapRecords = mapOnlyEpisodeId ? records : filteredRecords;
  const rawPoints = mapRecords.flatMap((record) => {
    const analysis = analysisByEpisode[String(record.episode.id)];
    if (!analysis) return [];
    return [{
      id: record.episode.id,
      rawX: Number(analysis.x),
      rawY: Number(analysis.y),
      topicId: analysis.dominant,
    }];
  });
  const extentPoints = allRecords.flatMap((record) => {
    const analysis = analysisByEpisode[String(record.episode.id)];
    return analysis ? [{ rawX: Number(analysis.x), rawY: Number(analysis.y) }] : [];
  });
  const extent = {
    minX: Math.min(...extentPoints.map((point) => point.rawX)), maxX: Math.max(...extentPoints.map((point) => point.rawX)),
    minY: Math.min(...extentPoints.map((point) => point.rawY)), maxY: Math.max(...extentPoints.map((point) => point.rawY)),
  };
  mapPoints = rawPoints.map((point) => ({
    ...point,
    x: padding + ((point.rawX - extent.minX) / Math.max(extent.maxX - extent.minX, 1e-9)) * width,
    y: padding + ((point.rawY - extent.minY) / Math.max(extent.maxY - extent.minY, 1e-9)) * height,
  }));
  for (const point of mapPoints) {
    const selected = point.id === selectedEpisodeId;
    context.beginPath();
    context.arc(point.x, point.y, selected ? 6 : 3.4, 0, Math.PI * 2);
    context.fillStyle = topicColour(point.topicId);
    context.globalAlpha = selected ? 1 : (mapOnlyEpisodeId ? 0.18 : 0.72);
    context.fill();
    if (selected) {
      context.strokeStyle = "#18372d";
      context.lineWidth = 2;
      context.stroke();
    }
  }
  context.globalAlpha = 1;
}

function nearestMapPoint(event) {
  const bounds = elements.map.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  let nearest;
  let distance = 12;
  for (const point of mapPoints) {
    const candidate = Math.hypot(point.x - x, point.y - y);
    if (candidate < distance) {
      nearest = point;
      distance = candidate;
    }
  }
  return nearest;
}

function renderEpisodeDetail(episodeId) {
  const episode = episodesById.get(Number(episodeId));
  const analysis = analysisByEpisode[String(episodeId)];
  if (!episode) return null;
  const container = document.createElement("div");
  container.className = "episode-detail";
  container.setAttribute("aria-live", "polite");
  container.addEventListener("click", (event) => event.stopPropagation());
  const descriptionTitle = document.createElement("h4");
  descriptionTitle.textContent = "Avsnittsbeskrivning";
  const fullDescription = document.createElement("p");
  fullDescription.className = "detail-full-description";
  fullDescription.textContent = episode.description || "Ingen beskrivning tillgänglig.";
  container.append(descriptionTitle, fullDescription);
  const wikipediaSpeakers = (episode.speakers || [])
    .map((qid) => speakersById[qid])
    .filter((speaker) => speaker?.wiki);
  if (wikipediaSpeakers.length) {
    const wikipedia = document.createElement("p");
    wikipedia.className = "detail-wikipedia";
    if (wikipediaSpeakers.length === 1) {
      wikipedia.append("Mer om värden på ");
      const link = document.createElement("a");
      link.href = wikipediaSpeakers[0].wiki;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Wikipedia";
      wikipedia.append(link);
    } else {
      wikipedia.append("Mer om värdarna på Wikipedia: ");
      wikipediaSpeakers.forEach((speaker, index) => {
        if (index) wikipedia.append(", ");
        const link = document.createElement("a");
        link.href = speaker.wiki;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = speaker.name;
        wikipedia.append(link);
      });
    }
    container.append(wikipedia);
  }
  const header = document.createElement("div");
  header.className = "detail-header";
  const title = document.createElement("h4");
  title.textContent = "Ämnen";
  header.append(title);
  const tags = document.createElement("div");
  tags.className = "detail-topics";
  for (const [topicId, share] of analysis?.topics || []) {
    const tag = document.createElement("button");
    tag.type = "button";
    tag.className = "topic-tag";
    tag.textContent = `${topicLabel(topicId)} ${Math.round(share * 100)} %`;
    tag.style.background = topicColour(topicId);
    tag.style.borderColor = topicColour(topicId);
    tag.style.color = "#fffdf9";
    tag.addEventListener("click", (event) => {
      event.stopPropagation();
      filterByTopic(topicId);
    });
    tags.append(tag);
  }
  container.append(header, tags);

  const sameSpeakerEpisodes = new Map();
  for (const qid of episode.speakers || []) {
    for (const record of records.filter((item) => item.episode.speakers?.includes(qid))) {
      if (record.episode.id === episode.id) continue;
      const season = seasonLabel(record.episode);
      if (!sameSpeakerEpisodes.has(season)) sameSpeakerEpisodes.set(season, record.episode);
    }
  }
  if (sameSpeakerEpisodes.size) {
    const sameTitle = document.createElement("h4");
    sameTitle.textContent = "Andra avsnitt med samma värdar";
    const sameList = document.createElement("div");
    sameList.className = "season-list";
    for (const [season, otherEpisode] of [...sameSpeakerEpisodes.entries()].sort(([, left], [, right]) => right.date.localeCompare(left.date))) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = season;
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        selectEpisode(otherEpisode.id);
        window.requestAnimationFrame(() => document.querySelector(`[data-episode-id="${otherEpisode.id}"]`)?.scrollIntoView({ block: "center" }));
      });
      sameList.append(button);
    }
    container.append(sameTitle, sameList);
  }

  const relatedTitle = document.createElement("h4");
  relatedTitle.textContent = "Liknande program";
  const relatedList = document.createElement("div");
  relatedList.className = "related-list";
  for (const [relatedId, similarity] of (analysis?.related || []).slice(0, 6)) {
    const related = episodesById.get(Number(relatedId));
    if (!related) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span></span><small>${Math.round(similarity * 100)} %</small>`;
    button.querySelector("span").textContent = `${displaySpeakers(related)} · ${seasonLabel(related)}`;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectEpisode(relatedId);
      window.requestAnimationFrame(() => document.querySelector(`[data-episode-id="${relatedId}"]`)?.scrollIntoView({ block: "center" }));
    });
    relatedList.append(button);
  }
  container.append(relatedTitle, relatedList);

  const metadataTitle = document.createElement("h4");
  metadataTitle.textContent = "Wikidata";
  const metadata = document.createElement("div");
  metadata.className = "metadata-bubbles";
  const metadataByKey = new Map();
  const occupationValues = new Set();
  const citizenshipValues = new Set();
  for (const [index, qid] of (episode.speakers || []).entries()) {
    const speaker = speakersById[qid];
    if (!speaker) continue;
    if (speaker.gender) metadataByKey.set(`gender:${speaker.gender}`, speaker.gender === "man" ? "Man" : speaker.gender === "kvinna" ? "Kvinna" : capitalise(speaker.gender));
    const age = episode.ages?.[index];
    if (Number.isFinite(age)) metadataByKey.set(`age:${age}`, `${age} år gammal vid programmet`);
    for (const occupation of speaker.occupations || []) occupationValues.add(occupation);
    for (const citizenship of speaker.citizenships || []) citizenshipValues.add(citizenship);
  }
  for (const label of metadataByKey.values()) {
    const bubble = document.createElement("span");
    bubble.className = "metadata-bubble";
    bubble.textContent = label;
    metadata.append(bubble);
  }
  if (occupationValues.size) {
    const label = document.createElement("span");
    label.className = "metadata-label metadata-section-label";
    label.textContent = "Sysselsättningar:";
    metadata.append(label);
    for (const value of occupationValues) {
      const bubble = document.createElement("span");
      bubble.className = "metadata-bubble";
      bubble.textContent = capitalise(value);
      metadata.append(bubble);
    }
  }
  if (citizenshipValues.size) {
    const label = document.createElement("span");
    label.className = "metadata-label metadata-section-label";
    label.textContent = "Medborgarskap:";
    metadata.append(label);
    for (const value of citizenshipValues) {
      const bubble = document.createElement("span");
      bubble.className = "metadata-bubble";
      bubble.textContent = value;
      metadata.append(bubble);
    }
  }
  if (metadata.children.length) container.append(metadataTitle, metadata);
  return container;
}

async function loadArchive() {
  const response = await fetch(EPISODES_URL);
  if (!response.ok) throw new Error("Kunde inte läsa programdata.");
  return response.json();
}

async function loadTopics() {
  try {
    const response = await fetch(TOPICS_URL);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

elements.filtersToggle.addEventListener("click", () => {
  setFiltersOpen(elements.filters.classList.contains("is-hidden"));
});

for (const input of [
  elements.search,
  elements.programType,
  elements.listenersHost,
  elements.sort,
  elements.returning,
  elements.gender,
  elements.citizenship,
  elements.occupation,
  elements.topic,
].filter(Boolean)) {
  input.addEventListener("input", scheduleRender);
  input.addEventListener("change", render);
}
elements.topic.addEventListener("change", renderTopicList);
elements.theme.addEventListener("change", () => {
  refreshLeafTopicOptions();
  renderTopicList();
  render();
});
elements.mapToggle.addEventListener("click", () => {
  const isOpen = !elements.mapDrawer.hidden;
  elements.mapDrawer.hidden = isOpen;
  elements.mapToggle.setAttribute("aria-expanded", String(!isOpen));
  elements.mapLabel.textContent = isOpen ? "Semantisk karta" : "Dölj semantisk karta";
  if (!isOpen) scheduleMapRender();
});
elements.map.addEventListener("pointermove", (event) => {
  const point = nearestMapPoint(event);
  elements.map.style.cursor = point ? "pointer" : "default";
  if (point) {
    const episode = episodesById.get(point.id);
    elements.map.title = `${displaySpeakers(episode)} · ${seasonLabel(episode)}`;
  } else {
    elements.map.removeAttribute("title");
  }
});
elements.map.addEventListener("click", (event) => {
  const point = nearestMapPoint(event);
  if (point) {
    selectEpisode(point.id);
  }
});
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
window.addEventListener("scroll", () => {
  scheduleVisibleRender();
  loadMoreNearPageEnd();
}, { passive: true });
window.addEventListener("resize", () => {
  buildVirtualLayout();
  scheduleVisibleRender();
  scheduleMapRender();
});
elements.reset.addEventListener("click", () => {
  setFilterControlsLocked(false);
  elements.search.value = "";
  elements.programType.value = "";
  if (elements.listenersHost) elements.listenersHost.checked = false;
  elements.yearFrom.value = yearBounds.min;
  elements.yearTo.value = yearBounds.max;
  updateYearRangeLabels();
  elements.sort.value = "newest";
  elements.returning.checked = false;
  elements.gender.value = "";
  elements.citizenship.value = "";
  elements.occupation.value = "";
  setSelectedValues(elements.theme, []);
  setSelectedValues(elements.topic, []);
  elements.ageFrom.value = ageBounds.min;
  elements.ageTo.value = ageBounds.max;
  mapOnlyEpisodeId = undefined;
  selectedEpisodeId = undefined;
  expandedDetailHeight = 0;
  updateAgeRangeLabels();
  renderTopicList();
  render();
});

try {
  const [archive, topicArchive] = await Promise.all([loadArchive(), loadTopics()]);
  const episodes = Array.isArray(archive?.episodes) ? archive.episodes : [];
  speakersById = archive?.speakers && typeof archive.speakers === "object"
    ? archive.speakers
    : {};
  topicsById = topicArchive?.topics && typeof topicArchive.topics === "object"
    ? topicArchive.topics
    : {};
  analysisByEpisode = topicArchive?.episodes && typeof topicArchive.episodes === "object"
    ? topicArchive.episodes
    : {};
  episodesById = new Map(episodes.map((episode) => [Number(episode.id), episode]));
  populateYearRange(episodes);
  populateAgeRange(episodes);
  populateSpeakerFilters();
  populateTopicFilters();
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
      topicIds: new Set(
        (analysisByEpisode[String(episode.id)]?.topics || []).map(([topicId]) => String(topicId)),
      ),
      topicParents: new Set(
        (analysisByEpisode[String(episode.id)]?.topics || [])
          .map(([topicId]) => topicsById[String(topicId)]?.parent)
          .filter(Boolean),
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
