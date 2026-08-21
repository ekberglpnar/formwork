# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

**formwork** — LLM'den yapılandırılmış çıktı alırken hangi alanların deterministik kural motoruna, hangilerinin modele ait olduğunu tek bir şemada bildirmeyi sağlayan Python kütüphanesi. Dilbilgisi tabanlı araçlar (Outlines, XGrammar) çıktının *biçimini* garanti eder; formwork çalışma zamanı bağlamına bağlı *anlamsal* kuralları garanti eder ve ihlal durumunda hedefli onarım yapar.

Ayrıntılı gerekçe ve rakip karşılaştırması `README.md`'de.

> Bu depo, üst dizindeki `/mnt/c/Users/EKBER/CLAUDE.md`'de anlatılan clinic-saas / SmartStock projeleriyle ilgisizdir. O dosya bu depo için geçerli değil.

## Ortam — önce bunu oku

Bu WSL kutusundaki sistem Python'ında **pip yok, `ensurepip` yok, parolasız sudo yok**. `python3 -m venv` çalışmaz. Ortam zaten kurulu; `.venv` mevcut ve `virtualenv` ile oluşturulmuş (`python3 -m venv` ile değil).

`.venv` silinir de yeniden kurulması gerekirse:

```bash
python3 -m virtualenv .venv          # venv DEĞİL — virtualenv kendi pip'ini taşır
.venv/bin/pip install -e ".[dev]"
```

`virtualenv` ve `pip` kullanıcının `~/.local` dizinine kurulu. Oradan da silinmişse `curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user --break-system-packages` ile geri getirilir. Sistem paketlerine dokunma.

## Komutlar

```bash
.venv/bin/python -m pytest -q                    # tüm testler (69 tane, <1 sn)
.venv/bin/python -m pytest tests/test_engine.py -q
.venv/bin/python -m pytest tests/test_engine.py::test_frozen_fields_survive_the_repair -q
.venv/bin/python -m pytest -q -k repair          # isme göre süzme

.venv/bin/ruff check src tests examples          # lint
.venv/bin/ruff check src tests examples --fix
.venv/bin/mypy                                   # strict; hedef paket pyproject'te tanımlı

.venv/bin/python examples/workout.py             # uçtan uca çalışan örnek
```

Değişiklikten sonra üçü de temiz olmalı — CI (`.github/workflows/ci.yml`) 3.11/3.12/3.13 üzerinde aynılarını ve örneği koşturur.

## Mimari

### Akış

```
spec sınıfı + ctx
   ↓ resolve_computed(ctx)        kural motoru: deterministik alanlar dolar
   ↓ model_facing_schema()        modele giden şema — computed alanlar YOK
   ↓ PromptBuilder.initial()      computed değerler "karar verilmiş olgu" olarak yazılır
   ↓ model
   ↓ şema doğrulaması             başarısız → structural_retry, tam yeniden üretim
   ↓ assemble()                   computed + model çıktısı → gerçek nesne
   ↓ spec.check(ctx)              anlamsal kurallar
       ├─ geçti  → sonuç
       ├─ kuralın repair'i var → deterministik onarım, model çağrısı YOK
       └─ yoksa → hedefli onarım: yalnızca ihlal edilen alanlar yeniden istenir
   ↓ deneme hakkı bitti → ConstraintError
```

### Taşıyıcı fikirler

**İki ayrı şema vardır.** `Spec` alt sınıfı tam nesneyi tanımlar; `model_facing_schema()` ise modele gönderilecek olanı `pydantic.create_model` ile dinamik kurar ve `computed` alanları dışarıda bırakır. Kütüphanenin temel iddiası buna dayanıyor: modelin göremediği alanı model yanlış yapamaz. Bir alanı modele açmak/kapatmak istiyorsan `fields.py`'deki rolünü değiştir, prompt'u değil.

**Roller `Annotated` metadata'sından okunur.** `computed()` / `chosen()` / `generated()` birer `FieldSpec` döndürür; `spec.py:_read_field_specs` bunları `model_fields[...].metadata` içinden toplar. Rol bildirmeyen alan `generated` sayılır. `Field(ge=1)` gibi Pydantic kısıtları modele giden şemaya taşınır (`_redeclare`) — bunlar ucuz yapısal korumalar, kaybedilmemeli.

**`Session` sans-IO'dur.** Durum makinesi istek üretir (`next_request`), çağıran yanıtı besler (`feed`). `generate` ve `agenerate` bunun üstünde birkaç satırlık sürücülerdir; test sahteleri de aynı döngüyü sürer. **Döngü mantığı yalnızca `Session` içinde yaşar** — sürücülere mantık ekleme, yoksa senkron ve asenkron yollar ayrışır.

**Onarım iki kademelidir.** `@rule(repair=...)` bildirilmişse `repair.py`'deki strateji model çağrısı olmadan düzeltir; strateji düzeltemediğinde `None` döner ve ihlal ayakta kalır. Kalan ihlaller için `_targeted_fields()` kuralların `fields=` ile işaret ettiği alanları toplar, şemayı ona daraltır, gerisini dondurur. `fields=` boş bırakılırsa hata verilmez — sessizce tam yeniden üretime düşülür (bkz. `test_rules_without_declared_fields_fall_back_to_everything`). Yeni kural yazarken `fields=` doğru vermek, hedefli onarımın çalışması için gereken tek şey.

**Kural dönüş tipleri normalize edilir.** Bir kural `None`/`True` (geçti), `False` (bildirilen mesajla düştü), `str`, `Violation`, ya da bunların iterable'ı döndürebilir; `Rule._normalise` hepsini `Violation` listesine çevirir ve tanınmayan tipte `TypeError` atar. İhlal mesajları **modele okutulmak üzere** yazılır — doğrudan onarım prompt'una girerler, o yüzden somut olsunlar (`"ex_42 kütüphanede yok"`, `"geçersiz değer"` değil).

### Bozmaman gereken değişmezler

- **`generate` asla kural ihlal eden nesne döndürmez.** Ya geçerli nesne ya istisna. `test_adversarial.py` bu özelliği 200 tohumla sınar; o dosya kütüphanenin varlık gerekçesidir.
- **`Attempt.ok` deterministik onarımı başarısızlık sayar.** Onarım koşmuşsa modelin çıktısı yanlıştı; `valid_first_try` kıyaslamada raporlanacak metrik olduğu için burayı gevşetmek kütüphanenin kendi başlık metriğini yalan yapar.
- **Sahte sağlayıcılar (`providers/fake.py`) genel API'nin parçasıdır**, test aracı değil. `Chaos`, `Recording`, `Scripted` README'de tanıtılıyor ve kullanıcıların kendi spec'lerini test etmesi için var.
- **Kod ve dokümanlar İngilizce.** Bilinçli bir karar: hedef kitle PyPI. (Kullanıcının diğer depoları Türkçe yorumludur; burada onu taklit etme.) Yorumlar *neden* böyle yapıldığını anlatır, *ne* yaptığını değil.

### Henüz yazılmamış olanlar

`README.md`'deki yol haritası şunları açıkça "yok" diye işaretliyor: gerçek sağlayıcı adaptörleri, `chosen` alanları için dilbilgisi arka ucu, ve **kıyaslama**. README verimlilik iddialarının ölçülmemiş olduğunu belirten bir uyarı taşıyor — kıyaslama gerçekten yazılana kadar o uyarıyı kaldırma.
