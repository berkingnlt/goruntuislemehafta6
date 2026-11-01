#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

struct YaricapAdayi
{
    int r;
    int oy;
};

int main()
{
    Mat orijinal = imread("images3.jpg", IMREAD_COLOR);

    if (orijinal.empty())
    {
        cout << "images3.jpg bulunamadi!" << endl;
        return -1;
    }

    // 1. Gonderilen goruntunun altindaki yazi seridini
    // disarida birak: sol ustten kare bolgeyi al.
    int kareBoyutu = std::min(orijinal.cols, orijinal.rows);

    Mat kare = orijinal(
        Rect(0, 0, kareBoyutu, kareBoyutu)
    ).clone();

    // Hesaplama maliyetini azalt.
    Mat resim;
    resize(kare, resim, Size(400, 400), 0, 0, INTER_AREA);

    double orijinaleOlcek = kareBoyutu / 400.0;

    Mat gri, kenarlar;
    cvtColor(resim, gri, COLOR_BGR2GRAY);
    Canny(gri, kenarlar, 50, 150);

    // Kenara olan uzaklik:
    // Kenar pikselinde 0, digerlerinde en yakin kenara mesafe.
    Mat tersKenarlar, kenaraUzaklik;
    bitwise_not(kenarlar, tersKenarlar);

    cv::distanceTransform(
        tersKenarlar,
        kenaraUzaklik,
        2,  // Öklid uzaklığı (DIST_L2)
        0   // Hassas hesaplama (DIST_MASK_PRECISE)
    );

    // 2. Sin/cos tablolari.
    const int aciAdimi = 4;
    const int aciSayisi = 360 / aciAdimi;

    vector<double> sinTablosu(aciSayisi);
    vector<double> cosTablosu(aciSayisi);

    for (int t = 0; t < aciSayisi; t++)
    {
        double aci = t * aciAdimi * CV_PI / 180.0;
        sinTablosu[t] = sin(aci);
        cosTablosu[t] = cos(aci);
    }

    // 3. Merkez adaylarini 4 piksel aralikla tara.
    // Herhangi bir merkez onceden verilmez.
    const int merkezAdimi = 4;

    int merkezXSayisi =
        (resim.cols + merkezAdimi - 1) / merkezAdimi;

    int merkezYSayisi =
        (resim.rows + merkezAdimi - 1) / merkezAdimi;

    const int minR = 12;
    const int maxR = 190;
    const int yaricapSayisi = maxR - minR + 1;

    // A[rIndeksi][merkezYIndeksi][merkezXIndeksi]
    size_t hucreSayisi =
        static_cast<size_t>(yaricapSayisi) *
        merkezYSayisi * merkezXSayisi;

    vector<int> oylama(hucreSayisi, 0);

    auto indeks = [=](int ri, int yi, int xi) -> size_t
        {
            return
                (static_cast<size_t>(ri) * merkezYSayisi + yi)
                * merkezXSayisi + xi;
        };

    cout << "3B oylama matrisi: "
        << yaricapSayisi << " x "
        << merkezYSayisi << " x "
        << merkezXSayisi << endl;

    // Sayisallastirma ve kenarin iki tarafini tolere eder.
    const float kenarToleransi = 1.5f;

    // Bir cizginin ic/dis kenarini iki ayri halka sayma.
    const int farkliHalkaAraligi = 12;

    // Sinir disindaki koordinatlar destek vermez.
    auto kenarDestegi = [&](int x, int y) -> bool
        {
            if (x < 0 || x >= resim.cols ||
                y < 0 || y >= resim.rows)
            {
                return false;
            }

            return kenaraUzaklik.at<float>(y, x)
                <= kenarToleransi;
        };

    // 4. ES MERKEZLILIGE OZEL OYLAMA.
    //
    // Her merkez ve aci icin:
    // - Ayni yaricapin iki karsi ucunda kenar ara.
    // - Bu kosulu saglayan farkli yaricaplari bul.
    // - En az iki ayri halka destekliyorsa oy ver.
    //
    // Tek bir cemberin destegi tek basina oy uretemez.
    vector<int> desteklenenYaricaplar;
    desteklenenYaricaplar.reserve(yaricapSayisi);

    for (int yi = 0; yi < merkezYSayisi; yi++)
    {
        int b = yi * merkezAdimi;

        for (int xi = 0; xi < merkezXSayisi; xi++)
        {
            int a = xi * merkezAdimi;

            for (int t = 0; t < aciSayisi; t++)
            {
                desteklenenYaricaplar.clear();

                for (int ri = 0; ri < yaricapSayisi; ri++)
                {
                    int r = minR + ri;

                    int x1 = cvRound(a + r * cosTablosu[t]);
                    int y1 = cvRound(b + r * sinTablosu[t]);

                    int x2 = cvRound(a - r * cosTablosu[t]);
                    int y2 = cvRound(b - r * sinTablosu[t]);

                    if (kenarDestegi(x1, y1) &&
                        kenarDestegi(x2, y2))
                    {
                        desteklenenYaricaplar.push_back(ri);
                    }
                }

                if (desteklenenYaricaplar.empty())
                    continue;

                // Liste yaricapa gore sirali.
                int ilk = desteklenenYaricaplar.front();
                int son = desteklenenYaricaplar.back();

                for (int ri : desteklenenYaricaplar)
                {
                    bool baskaHalkaDestegi =
                        ri - ilk >= farkliHalkaAraligi ||
                        son - ri >= farkliHalkaAraligi;

                    if (baskaHalkaDestegi)
                    {
                        // Ortak merkez + farkli yaricap destegi.
                        // Her acida bu hucreye en fazla bir oy.
                        oylama[indeks(ri, yi, xi)]++;
                    }
                }
            }
        }

        if (yi % 10 == 0)
        {
            cout << "Merkez satiri: "
                << yi << "/" << merkezYSayisi << endl;
        }
    }

    // 5. ORTAK merkez ve ona ait yaricaplari sec.
    // Ayri ayri bulunan cember merkezleri gruplanmaz.
    const int oyEsigi = 45;
    const int yaricapAyirma = 8;

    int enIyiSkor = -1;
    Point enIyiMerkez;
    vector<YaricapAdayi> enIyiYaricaplar;

    Mat merkezSkorlari = Mat::zeros(
        merkezYSayisi, merkezXSayisi, CV_32SC1
    );

    for (int yi = 0; yi < merkezYSayisi; yi++)
    {
        for (int xi = 0; xi < merkezXSayisi; xi++)
        {
            vector<YaricapAdayi> adaylar;

            for (int ri = 0; ri < yaricapSayisi; ri++)
            {
                int oy = oylama[indeks(ri, yi, xi)];

                if (oy >= oyEsigi)
                    adaylar.push_back({ minR + ri, oy });
            }

            sort(
                adaylar.begin(), adaylar.end(),
                [](const YaricapAdayi& a, const YaricapAdayi& b)
                {
                    if (a.oy != b.oy)
                        return a.oy > b.oy;

                    return a.r < b.r;
                }
            );

            // Ayni yaricap tepesinin yakin hucrelerini ele.
            vector<YaricapAdayi> secilenler;

            for (const auto& aday : adaylar)
            {
                bool yakinYaricapVar = false;

                for (const auto& secilen : secilenler)
                {
                    if (abs(aday.r - secilen.r) < yaricapAyirma)
                    {
                        yakinYaricapVar = true;
                        break;
                    }
                }

                if (!yakinYaricapVar)
                    secilenler.push_back(aday);
            }

            if (secilenler.size() < 2)
                continue;

            int skor = 0;

            for (const auto& secilen : secilenler)
                skor += secilen.oy;

            merkezSkorlari.at<int>(yi, xi) = skor;

            if (skor > enIyiSkor)
            {
                enIyiSkor = skor;
                enIyiMerkez = Point(
                    xi * merkezAdimi,
                    yi * merkezAdimi
                );

                enIyiYaricaplar = secilenler;
            }
        }
    }

    // 6. Orijinal boyuttaki goruntuye ciz.
    Mat sonuc = orijinal.clone();

    ofstream rapor("gorev5_cemberler.csv");
    rapor << "merkez_x,merkez_y,yaricap,oy\n";

    if (enIyiSkor >= 0)
    {
        sort(
            enIyiYaricaplar.begin(), enIyiYaricaplar.end(),
            [](const YaricapAdayi& a, const YaricapAdayi& b)
            {
                return a.r < b.r;
            }
        );

        Point merkez(
            cvRound(enIyiMerkez.x * orijinaleOlcek),
            cvRound(enIyiMerkez.y * orijinaleOlcek)
        );

        cout << "\nOrtak merkez: " << merkez << endl;
        cout << "Cember sayisi: "
            << enIyiYaricaplar.size() << endl;

        for (const auto& aday : enIyiYaricaplar)
        {
            int r = cvRound(aday.r * orijinaleOlcek);

            circle(
                sonuc, merkez, r,
                Scalar(0, 0, 255), 3, LINE_AA
            );

            cout << "Yaricap: " << r
                << " | Ortak destek oyu: "
                << aday.oy << endl;

            rapor << merkez.x << "," << merkez.y << ","
                << r << "," << aday.oy << "\n";
        }

        circle(
            sonuc, merkez, 5,
            Scalar(0, 255, 0), FILLED
        );
    }
    else
    {
        cout << "Es merkezli cember grubu bulunamadi."
            << endl;
    }

    rapor.close();

    // Ortak merkez skorlarini gorsellestir.
    Mat skorGoruntu;

    normalize(
        merkezSkorlari, skorGoruntu,
        0, 255, NORM_MINMAX, CV_8U
    );

    imwrite("gorev5_islenen_goruntu.png", resim);
    imwrite("gorev5_canny.png", kenarlar);
    imwrite("gorev5_ortak_merkez_oylari.png", skorGoruntu);
    imwrite("gorev5_es_merkezli_cemberler.png", sonuc);

    imshow("Canny - 400x400", kenarlar);

    namedWindow("Ortak Merkez Oylari", WINDOW_NORMAL);
    imshow("Ortak Merkez Oylari", skorGoruntu);

    namedWindow("Es Merkezli Cemberler", WINDOW_NORMAL);
    imshow("Es Merkezli Cemberler", sonuc);

    waitKey(0);
    return 0;
}