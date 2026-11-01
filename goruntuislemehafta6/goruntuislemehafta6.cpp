#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    Mat resim = imread("images1.jpg", IMREAD_COLOR);

    if (resim.empty())
    {
        cout << "images1.jpg bulunamadi!" << endl;
        return -1;
    }

    // 1. Griye donustur.
    Mat gri;
    cvtColor(resim, gri, COLOR_BGR2GRAY);

    // 2. Canny ile kenarlari bul.
    Mat kenarlar;
    Canny(gri, kenarlar, 50, 150, 3);

    // 3. Hocanin istedigi sirayla kenarlara median uygula.
    Mat medianKenarlar;
    medianBlur(kenarlar, medianKenarlar, 3);

    cout << "Canny kenar pikseli: "
        << countNonZero(kenarlar) << endl;

    cout << "Median sonrasi kenar pikseli: "
        << countNonZero(medianKenarlar) << endl;

    // 4. Standart Hough: rho ve theta degerlerini bulur.
    vector<Vec2f> dogrular;

    HoughLines(
        medianKenarlar,
        dogrular,
        1,              // Rho cozunurlugu: 1 piksel
        CV_PI / 180,    // Theta cozunurlugu: 1 derece
        30              // Oy esigi
    );

    Mat standartSonuc = resim.clone();

    double uzunluk = hypot(
        static_cast<double>(resim.cols),
        static_cast<double>(resim.rows)
    );

    for (size_t i = 0; i < dogrular.size(); i++)
    {
        double rho = dogrular[i][0];
        double theta = dogrular[i][1];

        double a = cos(theta);
        double b = sin(theta);

        double x0 = a * rho;
        double y0 = b * rho;

        Point baslangic(
            cvRound(x0 - uzunluk * b),
            cvRound(y0 + uzunluk * a)
        );

        Point bitis(
            cvRound(x0 + uzunluk * b),
            cvRound(y0 - uzunluk * a)
        );

        line(
            standartSonuc,
            baslangic,
            bitis,
            Scalar(0, 0, 255),
            2,
            LINE_AA
        );
    }

    // 5. Olasiliksal Hough: dogru parcalarinin uclarini bulur.
    vector<Vec4i> dogruParcalari;

    // Fonksiyon girdiyi degistirebilecegi icin kopyasini ver.
    Mat houghPGirdi = medianKenarlar.clone();

    HoughLinesP(
        houghPGirdi,
        dogruParcalari,
        1,              // Rho cozunurlugu
        CV_PI / 180,    // Theta cozunurlugu
        20,             // Oy esigi
        25,             // Minimum dogru parcasi uzunlugu
        10              // Birlesebilen parcalar arasindaki bosluk
    );

    Mat olasiliksalSonuc = resim.clone();

    for (size_t i = 0; i < dogruParcalari.size(); i++)
    {
        Vec4i d = dogruParcalari[i];

        line(
            olasiliksalSonuc,
            Point(d[0], d[1]),
            Point(d[2], d[3]),
            Scalar(0, 255, 0),
            2,
            LINE_AA
        );
    }

    cout << "\nHoughLines dogru sayisi: "
        << dogrular.size() << endl;

    cout << "HoughLinesP dogru parcasi sayisi: "
        << dogruParcalari.size() << endl;

    if (dogrular.empty() && dogruParcalari.empty())
    {
        cout << "\nDogru bulunamadi. Median ince Canny "
            << "kenarlarini silmis olabilir." << endl;
    }

    // 6. Sonuclari kaydet.
    imwrite("giris_canny.png", kenarlar);
    imwrite("giris_median.png", medianKenarlar);
    imwrite("giris_hough_lines.png", standartSonuc);
    imwrite("giris_hough_lines_p.png", olasiliksalSonuc);

    // 7. Goster.
    imshow("Orijinal", resim);
    imshow("Canny", kenarlar);
    imshow("Canny sonrasi Median", medianKenarlar);
    imshow("HoughLines - Kirmizi", standartSonuc);
    imshow("HoughLinesP - Yesil", olasiliksalSonuc);

    waitKey(0);
    return 0;
}