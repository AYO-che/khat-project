document.addEventListener("DOMContentLoaded", () => {
  let currentFile = null;    // الصورة الحالية
  let pendingFile = null;    // الصورة الجديدة المؤقتة إذا كان راه كاين صورة مسبقة

  const startUpload = document.getElementById('startUpload');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const imgUploadText = document.getElementById('imgUpload');
  const previewImg = document.getElementById('previewImg');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const resultCard = document.getElementById('resultCard');
  const uploadArea = document.querySelector('.upload-area');


  // عرض الصورة في الـ preview
  function handleFile(file) {
    if (!file) return;

    currentFile = file;

    const reader = new FileReader(); // FileReader يخلي المتصفح يقرا الصورة للـ preview

    reader.onload = e => {
      previewImg.src = e.target.result; // نعرض الصورة
      previewImg.classList.remove('d-none');
      imgUploadText.classList.add('d-none'); // نخفي النص القديم
      analyzeBtn.classList.remove('d-none'); // نظهر زر التحليل

      uploadArea.scrollIntoView({ behavior: 'smooth' }); // Scroll تلقائي للـ upload area
    };
    reader.readAsDataURL(file);
  }


  // ⭐ إدارة Modal إذا كان هناك صورة مسبقة
  function handleUploadClick() {
    if (currentFile) {
      // كاين صورة مسبقة → نطلع Modal
      const modal = new bootstrap.Modal(
        document.getElementById('confirmModal')
      );
      modal.show();
    } else {
      // ماكانش صورة → نفتح file picker مباشرة
      fileInput.click();
    }
  }


  // Click events على الزر و dropZone
  startUpload.addEventListener('click', handleUploadClick);
  dropZone.addEventListener('click', handleUploadClick);


  // عند اختيار ملف من file input
  fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
      pendingFile = e.target.files[0]; // نخزن الصورة الجديدة مؤقتًا
      handleFile(pendingFile);          // نعرض preview
    }
  });


  // Drag & Drop
  ['dragenter','dragover'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag');
    });
  });

  ['dragleave','drop'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag');
    });
  });

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      pendingFile = e.dataTransfer.files[0]; // نخزن الصورة الجديدة
      handleFile(pendingFile);             // نعرض preview
      e.dataTransfer.clearData();
    }
  });


  // زر "رفع صورة جديدة" داخل Modal
  document.getElementById('uploadNewBtn').addEventListener('click', () => {
    fileInput.click();
    if (pendingFile) {
      handleFile(pendingFile); // نستبدل الصورة الحالية بالجديدة
      pendingFile = null;
    }

    bootstrap.Modal
      .getInstance(document.getElementById('confirmModal'))
      .hide(); // إغلاق Modal
  });


  // التحليل الفوري
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    const form = new FormData();
    form.append("image", currentFile);

    analyzeBtn.textContent = '... جار التحليل';
    analyzeBtn.disabled = true;

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: form
      });

      if (!res.ok) throw new Error(`HTTP error ${res.status}`);

      const data = await res.json();

      analyzeBtn.textContent = 'النتيجة';
      analyzeBtn.disabled = false;

      // عرض النتيجة
      resultCard.classList.remove('d-none');
      document.querySelector('.score-box').textContent = data.confidence + "%";
      document.querySelector('.result-card h4').textContent =
        "جودة الخط: " + data.label;

      const ul = resultCard.querySelector('ul');
      ul.innerHTML = '';
      for (const [cls, val] of Object.entries(data.probabilities)) {
        const li = document.createElement('li');
        li.textContent = `${cls}: ${val}%`;
        ul.appendChild(li);
      }

      resultCard.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
      console.error(err);
      analyzeBtn.textContent = 'تحليل فوري';
      analyzeBtn.disabled = false;
      alert("حدث خطأ أثناء التحليل: " + err.message);
    }
  });
});
