"""
Мой тренер — обработка видео на GPU

Разворачивается на Modal: serverless-платформа, платит только за
секунды реальной работы. Простой контейнер не стоит ничего.

Стоимость на T4: $0.000164/сек. Матч на 10 минут обрабатывается
примерно за 3 минуты — около трёх центов. Бесплатных кредитов
($30/мес) хватает примерно на тысячу матчей.

Разворачивать из Colab, компьютер не нужен — см. deploy_modal.ipynb
"""

import modal

app = modal.App("moy-trener")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "ultralytics==8.3.0",
        "torch",
        "torchvision",
        "fastapi[standard]",
    )
)

# Веса YOLO кэшируются между запусками, чтобы не качать каждый раз
vol = modal.Volume.from_name("moy-trener-modeli", create_if_missing=True)
MODELI = "/modeli"

ALLOWED = [
    "https://crocus-health-mvp.github.io",
    "http://localhost:8000",
]


# ═══════════════════════════════════════════════════════════
#  РАЗБОР
# ═══════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="T4",
    volumes={MODELI: vol},
    timeout=900,          # потолок 15 минут на матч
    memory=8192,
)
def razobrat(video_bytes: bytes, imya: str = "match.mp4") -> dict:
    import cv2, os, json, tempfile
    import numpy as np

    os.environ["YOLO_CONFIG_DIR"] = MODELI

    rab = tempfile.mkdtemp()
    put = os.path.join(rab, "video.mp4")
    with open(put, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(put)
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if N < 10 or W < 100:
        return {"oshibka": "Не удалось прочитать видео"}

    # ── 1. Гистограммы кадров ──
    STEP = max(1, int(FPS // 5))

    def hist(fr):
        s = cv2.resize(fr, (160, 90))
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    cap = cv2.VideoCapture(put)
    hists, idx, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            hists.append(hist(fr))
            idx.append(i)
        i += 1
    cap.release()
    hists = np.array(hists, dtype="float32")

    # ── 2. Розыгрыши ──
    def seriya(m):
        dl, c = [], 0
        for v in m:
            if v:
                c += 1
            elif c:
                dl.append(c)
                c = 0
        if c:
            dl.append(c)
        return float(np.mean(dl)) if dl else 0.0

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, lbl, cent = cv2.kmeans(hists, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.flatten()

    ball = []
    for k in range(2):
        m = hists[lbl == k]
        if len(m) < 5:
            ball.append(-9)
            continue
        c = cent[k].astype("float32")
        bliz = np.mean([cv2.compareHist(c, h, cv2.HISTCMP_CORREL) for h in m[:500]])
        ball.append(bliz * 2 + seriya(lbl == k) / 50.0)
    igra = lbl == int(np.argmax(ball))

    a = igra.astype(np.uint8).reshape(1, -1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8))
    igra = a.flatten().astype(bool)

    rallies, start = [], None
    for j, p in enumerate(igra):
        if p and start is None:
            start = j
        elif not p and start is not None:
            t1, t2 = idx[start] / FPS, idx[j - 1] / FPS
            if t2 - t1 >= 2.5:
                rallies.append({"ot": round(t1, 1), "do": round(t2, 1),
                                "dlit": round(t2 - t1, 1)})
            start = None
    if start is not None:
        t1, t2 = idx[start] / FPS, idx[-1] / FPS
        if t2 - t1 >= 2.5:
            rallies.append({"ot": round(t1, 1), "do": round(t2, 1),
                            "dlit": round(t2 - t1, 1)})

    # ── 3. Игроки ──
    from ultralytics import YOLO
    ves = os.path.join(MODELI, "yolov8n.pt")
    model = YOLO(ves if os.path.exists(ves) else "yolov8n.pt")
    if not os.path.exists(ves):
        try:
            import shutil
            shutil.copy(model.ckpt_path, ves)
            vol.commit()
        except Exception:
            pass

    SAMPLE = max(1, int(FPS // 2))
    verh, niz = [], []

    cap = cv2.VideoCapture(put)
    for r in rallies:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["ot"] * FPS))
        konec = int(r["do"] * FPS)
        f = int(r["ot"] * FPS)
        tochki = []
        while f < konec:
            ok, fr = cap.read()
            if not ok:
                break
            if (f - int(r["ot"] * FPS)) % SAMPLE == 0:
                res = model(fr, classes=[0], conf=0.35, verbose=False)[0]
                lyudi = []
                for b in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b
                    if (y2 - y1) < H * 0.06:
                        continue
                    lyudi.append(((x1 + x2) / 2, y2))
                if len(lyudi) >= 2:
                    lyudi.sort(key=lambda p: p[1])
                    v, n = lyudi[0], lyudi[-1]
                    verh.append((v[0] / W, v[1] / H))
                    niz.append((n[0] / W, n[1] / H))
                    tochki.append({
                        "t": round(f / FPS, 1),
                        "verh": [round(v[0] / W, 3), round(v[1] / H, 3)],
                        "niz": [round(n[0] / W, 3), round(n[1] / H, 3)],
                    })
            f += 1
        r["tochki"] = tochki
    cap.release()

    # ── 4. Сводка ──
    def sredn(nabor):
        if not nabor:
            return None
        arr = np.array(nabor)
        return {
            "x": round(float(arr[:, 0].mean()), 3),
            "y": round(float(arr[:, 1].mean()), 3),
            "razbros_x": round(float(arr[:, 0].std()), 3),
            "razbros_y": round(float(arr[:, 1].std()), 3),
        }

    dl = [r["dlit"] for r in rallies]
    pauzy = [round(rallies[k + 1]["ot"] - rallies[k]["do"], 1)
             for k in range(len(rallies) - 1)]
    pauzy = [p for p in pauzy if 3 < p < 90]

    return {
        "versiya": 1,
        "video": imya,
        "dlitelnost_min": round(N / FPS / 60, 1),
        "itogo": {
            "rozygryshey": len(rallies),
            "srednyaya_dlina_sek": round(float(np.mean(dl)), 1) if dl else None,
            "mediana_sek": round(float(np.median(dl)), 1) if dl else None,
            "samyy_dlinnyy_sek": round(max(dl), 1) if dl else None,
            "dolya_igry_pct": round(sum(dl) / (N / FPS) * 100, 1) if dl else None,
            "srednyaya_pauza_sek": round(float(np.mean(pauzy)), 1) if pauzy else None,
        },
        "pozicii": {"dalniy": sredn(verh), "blizhniy": sredn(niz)},
        "rozygryshi": rallies,
    }



# ═══════════════════════════════════════════════════════════
#  ССЫЛКИ НА ОБЛАКА
#  Обычная ссылка на файл в Диске открывает страницу просмотра,
#  а не сам файл. Здесь она превращается в прямую.
#  Источник — только облако пользователя или корта.
# ═══════════════════════════════════════════════════════════

MAKS_SSYLKA_MB = 2048


def pryamaya_ssylka(url: str):
    """Возвращает (прямая_ссылка, откуда) либо (None, причина отказа)."""
    import re
    from urllib.parse import quote, urlparse

    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return None, "Ссылка должна начинаться с http"

    host = (urlparse(u).hostname or "").lower()

    # ── Ютуб и трансляции: не поддерживаем ──
    if any(x in host for x in ("youtube.", "youtu.be", "rutube.", "vimeo.", "twitch.")):
        return None, ("Видеохостинги не поддерживаются. Загрузи запись "
                      "в свой Диск и дай ссылку оттуда.")

    # ── Google Drive ──
    if "drive.google.com" in host or "docs.google.com" in host:
        m = (re.search(r"/file/d/([A-Za-z0-9_-]{10,})", u)
             or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", u))
        if not m:
            return None, "Не разобрал ссылку Google Drive"
        fid = m.group(1)
        return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t", "Google Drive"

    # ── Яндекс Диск ──
    if "disk.yandex" in host or "yadi.sk" in host:
        return ("https://cloud-api.yandex.net/v1/disk/public/resources/download"
                f"?public_key={quote(u, safe='')}"), "Яндекс Диск"

    # ── Dropbox ──
    if "dropbox.com" in host:
        base = u.split("?")[0]
        return base + "?dl=1", "Dropbox"

    # ── Прямая ссылка на файл ──
    if re.search(r"\.(mp4|mov|m4v|avi|mkv|webm)(\?|$)", u, re.I):
        return u, "прямая ссылка"

    return None, ("Не узнал источник. Поддерживаются Google Drive, "
                  "Яндекс Диск, Dropbox и прямые ссылки на файл.")


@app.function(image=image, timeout=1800, memory=8192)
def skachat_i_razobrat(url: str) -> dict:
    """Качает файл по ссылке и разбирает. Скачивание идёт на стороне
       Modal, поэтому размер не ограничен возможностями телефона."""
    import urllib.request, json, os

    pryamaya, otkuda = pryamaya_ssylka(url)
    if not pryamaya:
        return {"oshibka": otkuda}

    zapros = urllib.request.Request(pryamaya, headers={
        "User-Agent": "Mozilla/5.0 (compatible; moy-trener/1.0)",
        "Accept": "*/*",
    })

    try:
        with urllib.request.urlopen(zapros, timeout=120) as otvet:
            tip = (otvet.headers.get("Content-Type") or "").lower()

            # Яндекс отдаёт JSON со ссылкой на сам файл
            if "json" in tip:
                d = json.loads(otvet.read().decode())
                href = d.get("href")
                if not href:
                    return {"oshibka": "Яндекс Диск не отдал файл. "
                                       "Проверь, что ссылка публичная."}
                zapros2 = urllib.request.Request(href, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; moy-trener/1.0)"})
                with urllib.request.urlopen(zapros2, timeout=120) as o2:
                    danniye = o2.read(MAKS_SSYLKA_MB * 1024 * 1024)
            else:
                if "text/html" in tip:
                    return {"oshibka": "По ссылке пришла страница, а не файл. "
                                       "Проверь, что доступ открыт по ссылке."}
                danniye = otvet.read(MAKS_SSYLKA_MB * 1024 * 1024)
    except Exception as e:
        return {"oshibka": f"Не удалось скачать: {str(e)[:150]}"}

    if len(danniye) < 100000:
        return {"oshibka": "Файл слишком маленький — похоже, скачалась не запись, "
                           "а страница. Проверь доступ по ссылке."}

    imya = pryamaya.split("/")[-1].split("?")[0][:60] or "video.mp4"
    rez = razobrat.remote(danniye, imya)
    if isinstance(rez, dict):
        rez["istochnik"] = otkuda
        rez["razmer_mb"] = round(len(danniye) / 1048576, 1)
    return rez


# ═══════════════════════════════════════════════════════════
#  ВЕБ-ЭНДПОИНТ
#  Приложение шлёт сюда видео, получает готовый разбор.
# ═══════════════════════════════════════════════════════════

MAKS_MB = 220


@app.function(image=image, timeout=900)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    api = FastAPI(docs_url=None, redoc_url=None)

    api.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/")
    def zhiv():
        return {"status": "ok", "servis": "moy-trener", "maks_mb": MAKS_MB}

    @api.post("/po-ssylke")
    async def po_ssylke(request: Request):
        """Ставит задачу в фон и сразу отдаёт её номер.
           Скачивание и разбор идут дольше, чем живёт HTTP-соединение."""
        try:
            telo = await request.json()
        except Exception:
            return JSONResponse({"oshibka": "Ожидается JSON с полем url"}, 400)

        url = (telo or {}).get("url", "")
        if not url:
            return JSONResponse({"oshibka": "Пустая ссылка"}, 400)

        pryamaya, otkuda = pryamaya_ssylka(url)
        if not pryamaya:
            return JSONResponse({"oshibka": otkuda}, 400)

        vyzov = skachat_i_razobrat.spawn(url)
        return JSONResponse({"nomer": vyzov.object_id, "istochnik": otkuda})

    @api.post("/analiz")
    async def analiz(request: Request):
        """То же для файла, загруженного прямо из приложения."""
        danniye = await request.body()
        if not danniye:
            return JSONResponse({"oshibka": "Пустой запрос"}, 400)

        if len(danniye) > MAKS_MB * 1024 * 1024:
            return JSONResponse(
                {"oshibka": f"Файл больше {MAKS_MB} МБ. "
                            "Залей в Диск и дай ссылку — там ограничения нет."}, 413)

        imya = request.headers.get("x-file-name", "match.mp4")
        vyzov = razobrat.spawn(danniye, imya)
        return JSONResponse({"nomer": vyzov.object_id})

    @api.get("/status/{nomer}")
    def status(nomer: str):
        """Готово ли. Пока считается — отдаёт 'в работе'."""
        try:
            vyzov = modal.FunctionCall.from_id(nomer)
        except Exception:
            return JSONResponse({"oshibka": "Задача не найдена"}, 404)

        try:
            rez = vyzov.get(timeout=0)
        except TimeoutError:
            return JSONResponse({"gotovo": False})
        except Exception as e:
            return JSONResponse({"gotovo": True, "oshibka": str(e)[:200]})

        if isinstance(rez, dict) and rez.get("oshibka"):
            return JSONResponse({"gotovo": True, "oshibka": rez["oshibka"]})
        return JSONResponse({"gotovo": True, "razbor": rez})

    return api


# Локальный прогон для проверки: modal run modal_app.py --put video.mp4
@app.local_entrypoint()
def main(put: str = ""):
    import json
    if not put:
        print("Укажи файл: modal run modal_app.py --put video.mp4")
        return
    with open(put, "rb") as f:
        rez = razobrat.remote(f.read(), put.split("/")[-1])
    print(json.dumps(rez.get("itogo", rez), ensure_ascii=False, indent=2))
