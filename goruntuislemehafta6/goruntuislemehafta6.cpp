#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

struct Cember
{
    int x;
    int y;
    int r;
    int oy;
};

int main()
{
    Mat resim = imread("images2.jpg", IMREAD_COLOR);

    if (resim.empty())
    {
        cout << "images2.jpg bulunamadi!" << endl;
        return -1;
    }

    // 1. Kenarlari hazirla.
    // Bu maddede Canny SONRASI median sarti yok.
    // Paralarin ic desenlerini azaltmak icin griyi filtrele.
    Mat gri, filtreli, kenarlar;

    cvtColor(resim, gri, COLOR_BGR2GRAY);
    GaussianBlur(gri, filtreli, Size(5, 5), 1.5);
    Canny(filtreli, kenarlar, 80, 160);

    int genislik = resim.cols;
    int yukseklik = resim.rows;

    // Gonderdigin 554x554 para goruntusu icin.
    const int minR = 30;
    const int maxR = 70;
    const int yaricapSayisi = maxR - minR + 1;

    // 2 derecelik adimla 180 farkli aci.
    const int aciAdimi = 2;
    const int aciSayisi = 360 / aciAdimi;

    // 2. Sin/cos degerlerini bastan hesaplayip sakla.
    vector<double> sinTablosu(aciSayisi);
    vector<double> cosTablosu(aciSayisi);

    for (int t = 0; t < aciSayisi; t++)
    {
        double radyan = t * aciAdimi * CV_PI / 180.0;

        sinTablosu[t] = sin(radyan);
        cosTablosu[t] = cos(radyan);
    }

    // 3. Uc boyutlu matris: A[rIndeksi][merkezY][merkezX].
    // Bellekte tek vector icinde tutuluyor.
    size_t hucreSayisi =
        static_cast<size_t>(yaricapSayisi) *
        yukseklik * genislik;

    vector<int> oylama(hucreSayisi, 0);

    auto indeks = [=](int rIndeksi, int y, int x) -> size_t
        {
            return
                (static_cast<size_t>(rIndeksi) * yukseklik + y)
                * genislik + x;
        };

    cout << "3B matris: "
        << yaricapSayisi << " x "
        << yukseklik << " x "
        << genislik << endl;

    cout << "Matris bellegi: "
        << hucreSayisi * sizeof(int) / (1024.0 * 1024.0)
        << " MB" << endl;

    vector<Point> kenarNoktalari;
    findNonZero(kenarlar, kenarNoktalari);

    cout << "Kenar pikseli: "
        << kenarNoktalari.size() << endl;

    // 4. OYLAMA:
    // x = a + r*cos(theta)
    // y = b + r*sin(theta)
    //
    // Aday merkez:
    // a = x - r*cos(theta)
    // b = y - r*sin(theta)
    for (int ri = 0; ri < yaricapSayisi; ri++)
    {
        int r = minR + ri;

        for (size_t p = 0; p < kenarNoktalari.size(); p++)
        {
            int x = kenarNoktalari[p].x;
            int y = kenarNoktalari[p].y;

            for (int t = 0; t < aciSayisi; t++)
            {
                int a = cvRound(x - r * cosTablosu[t]);
                int b = cvRound(y - r * sinTablosu[t]);

                if (a < 0 || a >= genislik ||
                    b < 0 || b >= yukseklik)
                {
                    continue;
                }

                // A[r][b][a] bir oy alir.
                int& hucre = oylama[indeks(ri, b, a)];
                hucre++;

                if (ri == 0 && p < 2 &&
                    (t == 0 || t == 45))
                {
                    cout << "Kenar (" << x << "," << y << ")"
                        << " r=" << r
                        << " aci=" << t * aciAdimi
                        << " -> merkez=(" << a << "," << b
                        << ") oy=" << hucre << endl;
                }
            }
        }

        cout << "Oylama tamamlandi: r=" << r << endl;
    }

    // 5. Esigi gecen 3B yerel maksimumlari bul.
    const int oyEsigi = 45;

    vector<Cember> adaylar;

    // Gorsellestirme icin her merkezdeki en yuksek
    // yaricap oyunu al: P(y,x) = max_r A(r,y,x).
    Mat merkezOylari = Mat::zeros(
        yukseklik, genislik, CV_32SC1
    );

    for (int ri = 0; ri < yaricapSayisi; ri++)
    {
        for (int y = 0; y < yukseklik; y++)
        {
            for (int x = 0; x < genislik; x++)
            {
                int oy = oylama[indeks(ri, y, x)];

                int& enYuksek = merkezOylari.at<int>(y, x);
                enYuksek = std::max(enYuksek, oy);

                if (oy < oyEsigi)
                    continue;

                bool maksimum = true;
                size_t mevcutIndeks = indeks(ri, y, x);

                for (int dr = -1; dr <= 1 && maksimum; dr++)
                {
                    int nr = ri + dr;

                    if (nr < 0 || nr >= yaricapSayisi)
                        continue;

                    for (int dy = -1; dy <= 1 && maksimum; dy++)
                    {
                        int ny = y + dy;

                        if (ny < 0 || ny >= yukseklik)
                            continue;

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int nx = x + dx;

                            if (nx < 0 || nx >= genislik)
                                continue;

                            size_t komsuIndeks = indeks(nr, ny, nx);
                            int komsuOy = oylama[komsuIndeks];

                            if (komsuOy > oy ||
                                (komsuOy == oy &&
                                    komsuIndeks < mevcutIndeks))
                            {
                                maksimum = false;
                                break;
                            }
                        }
                    }
                }

                if (maksimum)
                    adaylar.push_back({ x, y, minR + ri, oy });
            }
        }
    }

    // 6. En yuksek oy alan adaylari once degerlendir.
    sort(
        adaylar.begin(),
        adaylar.end(),
        [](const Cember& a, const Cember& b)
        {
            return a.oy > b.oy;
        }
    );

    // Ayni paranin ic/dis kenarlarini tekrar secmeyi azalt.
    // Bu madde ayri merkezli paralar icindir.
    // Es merkezli cemberler icin bu secim kullanilmayacak.
    const double minimumMerkezMesafesi = 70.0;

    vector<Cember> bulunanlar;

    for (const Cember& aday : adaylar)
    {
        bool yakinMerkezVar = false;

        for (const Cember& secilen : bulunanlar)
        {
            double mesafe = hypot(
                static_cast<double>(aday.x - secilen.x),
                static_cast<double>(aday.y - secilen.y)
            );

            if (mesafe < minimumMerkezMesafesi)
            {
                yakinMerkezVar = true;
                break;
            }
        }

        if (!yakinMerkezVar)
            bulunanlar.push_back(aday);
    }

    // 7. Cemberleri orijinal resme ciz.
    Mat sonuc = resim.clone();
    ofstream rapor("gorev4_cemberler.csv");
    rapor << "merkez_x,merkez_y,yaricap,oy\n";

    for (size_t i = 0; i < bulunanlar.size(); i++)
    {
        const Cember& c = bulunanlar[i];
        Point merkez(c.x, c.y);

        circle(
            sonuc, merkez, c.r,
            Scalar(0, 0, 255), 2, LINE_AA
        );

        circle(
            sonuc, merkez, 3,
            Scalar(0, 255, 0), FILLED
        );

        putText(
            sonuc, to_string(i + 1),
            Point(c.x + 5, c.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 0, 0), 2
        );

        cout << "Cember " << i + 1
            << ": merkez=(" << c.x << "," << c.y << ")"
            << " r=" << c.r
            << " oy=" << c.oy << endl;

        rapor << c.x << "," << c.y << ","
            << c.r << "," << c.oy << "\n";
    }

    rapor.close();

    // 8. Oylama projeksiyonunu goruntu olarak kaydet.
    Mat oylamaFloat, oylamaGoruntu;

    merkezOylari.convertTo(oylamaFloat, CV_32F);
    oylamaFloat += 1.0f;
    cv::log(oylamaFloat, oylamaFloat);

    normalize(
        oylamaFloat, oylamaGoruntu,
        0, 255, NORM_MINMAX, CV_8U
    );

    imwrite("gorev4_canny.png", kenarlar);
    imwrite("gorev4_oylama_projeksiyonu.png", oylamaGoruntu);
    imwrite("gorev4_manuel_cemberler.png", sonuc);

    double maksimumOy;
    minMaxLoc(merkezOylari, nullptr, &maksimumOy);

    cout << "\nEn yuksek oy: " << maksimumOy << endl;
    cout << "Oy esigi: " << oyEsigi << endl;
    cout << "Bulunan cember: " << bulunanlar.size() << endl;

    imshow("Canny", kenarlar);
    imshow("Oylama - Yaricaplar Uzerinde Maksimum", oylamaGoruntu);
    imshow("Manuel Hough Cemberler", sonuc);

    waitKey(0);
    return 0;
}