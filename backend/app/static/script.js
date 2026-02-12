
const DEFAULT_REQUEST = {
  question: "Trước khi có chữ Nôm, người Việt có từng có chữ viết riêng không?",
  top_k: 10,
  pool_size: 20,
  rerank: true,
};

let currentRequest = { ...DEFAULT_REQUEST };
let selectedSource = null;
let currentAnswer = null;

const submitButton = document.getElementById("submit-button");
const questionInput = document.getElementById("question");
const loadingIndicator = document.getElementById("loading-indicator");
const errorText = document.getElementById("error-text");
const answerContainer = document.getElementById("answer-container");
const answerText = document.getElementById("answer-text");
const citationListContainer = document.getElementById("citation-list-container");
const buttonText = document.getElementById("button-text");
const buttonIcon = document.querySelector(".button-icon");
const answerCard = document.getElementById("answer-card");

questionInput.value = currentRequest.question;

questionInput.addEventListener("input", (e) => {
  currentRequest.question = e.target.value;
  answerCard.classList.remove("answer-card--highlight");
});

submitButton.addEventListener("click", handleSubmit);

async function handleSubmit() {
  setLoading(true);
  setError(null);
  selectedSource = null;
  expandedKey = null;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(currentRequest),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    currentAnswer = data;
    selectedSource = data.sources?.[0] ?? null;

    displayAnswer(data);
    
    // Highlight answer card
    answerCard.classList.add("answer-card--highlight");
    // Scroll to answer
    setTimeout(() => {
        answerCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 100);

  } catch (err) {
    setError(err.message || "Failed to reach backend");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  if (isLoading) {
    submitButton.classList.add("is-loading");
    loadingIndicator.style.display = "flex";
    buttonIcon.style.display = "none";
    buttonText.textContent = "Đang phân tích…";
    answerContainer.style.display = "none";
    errorText.style.display = "none";
  } else {
    submitButton.classList.remove("is-loading");
    loadingIndicator.style.display = "none";
    buttonIcon.style.display = "inline-flex";
    buttonText.textContent = "Đặt câu hỏi";
  }
}

function setError(message) {
  if (message) {
    errorText.textContent = message;
    errorText.style.display = "block";
  } else {
    errorText.style.display = "none";
  }
}

function displayAnswer(data) {
  answerText.textContent = data.answer;
  answerContainer.style.display = "block";
  renderCitations(data.sources);
}

let expandedKey = null;

function renderCitations(sources) {
  citationListContainer.innerHTML = "";
  if (!sources || sources.length === 0) return;

  const header = document.createElement("strong");
  header.textContent = "Nguồn tham chiếu";
  citationListContainer.appendChild(header);

  const resolvedSource = selectedSource || sources[0];

  sources.forEach((source, idx) => {
    const key = `${source.file_name ?? source.label}-${source.page_number ?? idx}`;
    const isActive = (resolvedSource?.viewer_url === source.viewer_url && resolvedSource?.label === source.label) || selectedSource?.label === source.label;
    const isExpanded = expandedKey === key;

    const article = document.createElement("article");
    article.className = `citation-item${isActive ? " active" : ""}${isExpanded ? " expanded" : ""}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation-toggle";
    button.onclick = () => {
        selectedSource = source;
        expandedKey = isExpanded ? null : key;
        answerCard.classList.remove("answer-card--highlight");
        renderCitations(sources);
    };

    const metaDiv = document.createElement("div");
    metaDiv.className = "citation-meta";

    const titleSpan = document.createElement("span");
    titleSpan.className = "citation-title";
    
    // Title logic
    const titleParts = [];
    const bookTitle = source.book_title?.trim();
    const cleanedChapter = source.chapter ? source.chapter.replace(/_/g, " ").replace(/\s+/g, " ").trim() : "";
    const hideChapter = bookTitle === "Ngôn ngữ. Văn tự. Ngữ văn (Tuyển tập)";

    if (bookTitle) titleParts.push(bookTitle);
    if (cleanedChapter && !hideChapter && cleanedChapter !== bookTitle && !titleParts.includes(cleanedChapter)) {
        titleParts.push(cleanedChapter);
    }
    const titleText = titleParts.length > 0 ? titleParts.join(" – ") : source.label;
    titleSpan.textContent = titleText;

    metaDiv.appendChild(titleSpan);

    if (source.page_number) {
        const pageSpan = document.createElement("span");
        pageSpan.className = "citation-page";
        pageSpan.textContent = `P.${source.page_number}`;
        metaDiv.appendChild(pageSpan);
    }

    button.appendChild(metaDiv);

    const iconSpan = document.createElement("span");
    iconSpan.className = `citation-icon${isExpanded ? " citation-icon--open" : ""}`;
    iconSpan.ariaHidden = "true";
    button.appendChild(iconSpan);

    article.appendChild(button);

    if (isExpanded) {
        const snippetDiv = document.createElement("div");
        snippetDiv.className = "snippet";
        snippetDiv.textContent = source.text;
        article.appendChild(snippetDiv);
    }

    citationListContainer.appendChild(article);
  });
}

answerCard.addEventListener("click", () => {
    answerCard.classList.remove("answer-card--highlight");
});
