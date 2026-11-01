#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

struct Dogru
{
    int rho;
    int theta;
    int oy;
};

int main()
{
    Mat resim = imread("images1.jpg", IMREAD_COLOR);

    if (resim.empty())
    {
        cout << "images1.jpg bulunamadi!" << endl;
        return -1;
    }

    // 1. Gri goruntu ve Canny kenarlari.
    Mat gri, kenarlar;
    cvtColor(resim, gri, COLOR_BGR2GRAY);
    Canny(gri, kenarlar, 50, 150, 3);

    // 2. Sin ve cos degerlerini bir kez hesaplayip sakla.
    const int aciSayisi = 180;

    vector<double> sinTablosu(aciSayisi);
    vector<double> cosTablosu(aciSayisi);

    for (int theta = 0; theta < aciSayisi; theta++)
    {
        double radyan = theta * CV_PI / 180.0;

        sinTablosu[theta] = sin(radyan);
        cosTablosu[theta] = cos(radyan);
    }

    // Rho negatif de olabilir.
    // Aralik: [-kosegen, +kosegen]
    int kosegen = static_cast<int>(ceil(hypot(
        static_cast<double>(resim.cols),
        static_cast<double>(resim.rows)
    )));

    int rhoSayisi = 2 * kosegen + 1;

    // 3. Oylama matrisi: satir = rho, sutun = theta.
    Mat oylama = Mat::zeros(
        rhoSayisi,
        aciSayisi,
        CV_32SC1
    );

    cout << "Oylama matrisi: "
        << rhoSayisi << " x " << aciSayisi << endl;

    cout << "Kenar pikseli sayisi: "
        << countNonZero(kenarlar) << endl;

    int ornekPikselSayisi = 0;

    // 4. Her kenar pikseli her aci icin oy verir.
    for (int y = 0; y < kenarlar.rows; y++)
    {
        for (int x = 0; x < kenarlar.cols; x++)
        {
            if (kenarlar.at<uchar>(y, x) == 0)
                continue;

            for (int theta = 0; theta < aciSayisi; theta++)
            {
                // Dogru denklemi:
                // rho = x*cos(theta) + y*sin(theta)
                int rho = cvRound(
                    x * cosTablosu[theta] +
                    y * sinTablosu[theta]
                );

                // Negatif rho degerini gecerli satira tasi.
                int rhoIndeksi = rho + kosegen;

                // OYLAMA ISLEMI:
                oylama.at<int>(rhoIndeksi, theta)++;

                // Ilk 3 kenar pikselinin ornek oylarini goster.
                if (ornekPikselSayisi < 3 &&
                    (theta == 0 || theta == 45 || theta == 90))
                {
                    cout << "Piksel (" << x << ", " << y << ")"
                        << " theta=" << theta
                        << " rho=" << rho
                        << " -> matris[" << rhoIndeksi
                        << "][" << theta << "] = "
                        << oylama.at<int>(rhoIndeksi, theta)
                        << endl;
                }
            }

            ornekPikselSayisi++;
        }
    }

    // 5. Ham oylama degerlerini CSV olarak kaydet.
    ofstream csv("gorev2_oylama.csv");

    if (!csv)
    {
        cout << "CSV dosyasi olusturulamadi!" << endl;
        return -1;
    }

    csv << "rho";

    for (int theta = 0; theta < aciSayisi; theta++)
        csv << "," << theta;

    csv << "\n";

    for (int r = 0; r < rhoSayisi; r++)
    {
        csv << r - kosegen;

        for (int theta = 0; theta < aciSayisi; theta++)
            csv << "," << oylama.at<int>(r, theta);

        csv << "\n";
    }

    csv.close();

    // 6. Oylama matrisini goruntuye donustur.
    // Logaritma, dusuk oylarin da gorunmesini saglar.
    // Esikleme asagida HAM matris uzerinde yapilir.
    Mat oylamaFloat;
    oylama.convertTo(oylamaFloat, CV_32F);

    oylamaFloat += 1.0f;
    cv::log(oylamaFloat, oylamaFloat);

    Mat oylamaGoruntu;
    normalize(
        oylamaFloat,
        oylamaGoruntu,
        0,
        255,
        NORM_MINMAX,
        CV_8U
    );

    imwrite("gorev2_oylama_matrisi.png", oylamaGoruntu);

    double enYuksekOy;
    minMaxLoc(oylama, nullptr, &enYuksekOy);

    cout << "\nEn yuksek oy: " << enYuksekOy << endl;

    // 7. Esigi gecen yerel maksimumlari bul.
    // Komsuluk kontrolu, ayni dogrunun yakin hucrelerden
    // tekrar tekrar secilmesini azaltir.
    const int oyEsigi = 70;
    const int rhoKomsuluk = 4;
    const int aciKomsuluk = 2;

    vector<Dogru> dogrular;

    for (int r = 0; r < rhoSayisi; r++)
    {
        for (int theta = 0; theta < aciSayisi; theta++)
        {
            int oy = oylama.at<int>(r, theta);

            if (oy < oyEsigi)
                continue;

            bool yerelMaksimum = true;

            for (int dr = -rhoKomsuluk;
                dr <= rhoKomsuluk && yerelMaksimum;
                dr++)
            {
                for (int dt = -aciKomsuluk;
                    dt <= aciKomsuluk;
                    dt++)
                {
                    if (dr == 0 && dt == 0)
                        continue;

                    int komsuR = r + dr;
                    int komsuTheta = theta + dt;

                    if (komsuR < 0 || komsuR >= rhoSayisi)
                        continue;

                    // 0 ve 180 derece sinirinda
                    // (rho, theta) = (-rho, theta + 180).
                    if (komsuTheta < 0)
                    {
                        komsuTheta += aciSayisi;
                        komsuR = 2 * kosegen - komsuR;
                    }
                    else if (komsuTheta >= aciSayisi)
                    {
                        komsuTheta -= aciSayisi;
                        komsuR = 2 * kosegen - komsuR;
                    }

                    int komsuOy =
                        oylama.at<int>(komsuR, komsuTheta);

                    // Esit oy durumunda tek hucreyi sec.
                    bool oncekiHucre =
                        komsuR < r ||
                        (komsuR == r && komsuTheta < theta);

                    if (komsuOy > oy ||
                        (komsuOy == oy && oncekiHucre))
                    {
                        yerelMaksimum = false;
                        break;
                    }
                }
            }

            if (yerelMaksimum)
            {
                dogrular.push_back({
                    r - kosegen,
                    theta,
                    oy
                    });
            }
        }
    }

    // 8. Bulunan dogrulari orijinal goruntuye ciz.
    Mat sonuc = resim.clone();

    for (const Dogru& dogru : dogrular)
    {
        double a = cosTablosu[dogru.theta];
        double b = sinTablosu[dogru.theta];

        double x0 = dogru.rho * a;
        double y0 = dogru.rho * b;

        Point p1(
            cvRound(x0 - kosegen * b),
            cvRound(y0 + kosegen * a)
        );

        Point p2(
            cvRound(x0 + kosegen * b),
            cvRound(y0 - kosegen * a)
        );

        line(
            sonuc,
            p1,
            p2,
            Scalar(0, 0, 255),
            2,
            LINE_AA
        );

        cout << "Dogru: rho=" << dogru.rho
            << " theta=" << dogru.theta
            << " oy=" << dogru.oy << endl;
    }

    cout << "\nOy esigi: " << oyEsigi << endl;
    cout << "Bulunan dogru sayisi: "
        << dogrular.size() << endl;

    imwrite("gorev2_canny.png", kenarlar);
    imwrite("gorev2_manuel_hough.png", sonuc);

    imshow("Orijinal", resim);
    imshow("Canny", kenarlar);

    namedWindow("Oylama Matrisi", WINDOW_NORMAL);
    imshow("Oylama Matrisi", oylamaGoruntu);

    imshow("Manuel Hough - Kirmizi Dogrular", sonuc);

    waitKey(0);
    return 0;
}