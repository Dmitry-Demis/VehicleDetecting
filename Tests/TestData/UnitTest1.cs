using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Security.Cryptography.X509Certificates;
using Helper;
using AForge.Imaging;
using AForge.Imaging.Filters;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using TesseractSharp;
using Image = AForge.Imaging.Image;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace TestData
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void EmptyOrIsVehicle()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Train\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            VehicleAvailability vehicle = new($@"..\..\..\..\..\Helper\Models\carEmpty.h5");
            foreach (var file in files)
            {
                var @class = int.Parse(Path.GetFileName(file)[0].ToString());
                var result = vehicle.RecognizeVehicle(new Bitmap(file));
                Assert.AreEqual(result, @class == 1);
            }
        }

        [TestMethod]
        public void FilteredFunction()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            foreach (var t in files)
            {
                Bitmap? bitmap = null;
                using (var fs = File.Open(t, FileMode.Open)) bitmap = new Bitmap(fs);
                if (File.Exists(t)) File.Delete(t);
                GaussianSharpen sharpen = new();
                bitmap = sharpen.Apply(bitmap);
                BrightnessCorrection brightness = new(50);
                bitmap = brightness.Apply(bitmap);
               // bitmap = sharpen.Apply(bitmap);
                //bitmap = brightness.Apply(bitmap);
                bitmap.Save(t);
            }
        }
        [TestMethod]
        public void RecognizeNumber()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            for (int i = 0; i < files.Length; i++)
            {
                var gray = new Mat(files[i], ImreadModes.Grayscale);
                var coeff = 10;
                Cv2.Resize(gray, gray, new Size(), fx: coeff, fy: coeff, InterpolationFlags.Cubic);
                Mat thresh = new();
                Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.BinaryInv);
                Mat rect_kern = new();
                rect_kern =  Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5,5)); 
                Mat dilation = new();
                Cv2.Dilate(thresh, dilation, rect_kern);
              Cv2.FindContours(dilation, out var contours, out var hierarchy, RetrievalModes.Tree,
                 ContourApproximationModes.ApproxSimple);
                Array.Sort(contours, ((points, points1) =>
                {
                    var l = Cv2.BoundingRect(points);
                    var r = Cv2.BoundingRect(points1);
                    return (l.X > r.X) ? 1 : -1;
                }));
                List<Point[]> mats = new ();
                //Cv2.DrawContours(gray, new List<Point[]> { contours[0] }, -1, Scalar.Black);
               // Cv2.ImShow($"{i}{nameof(thresh)}", gray);
                //Cv2.WaitKey();
                for (var index1 = 0; index1 < contours.Length; index1++)
                {
                    var contour = contours[index1];
                    var rect = Cv2.BoundingRect(contour);
                    if ((1.0 * thresh.Height) / rect.Height > 6)
                    {
                        continue;
                    }

                    double ratio = (1.0 * rect.Height) / rect.Width;
                    if (ratio < 1.3)
                    {
                        continue;
                    }

                    if ((1.0 * thresh.Width) / rect.Width > 12)
                    {
                        continue;
                    }

                    var area = Cv2.ContourArea(contour);
                    if (area < 100)
                    {
                        continue;
                    }
                    mats.Add(contour);
                }

                Cv2.DrawContours(gray, mats, -1, Scalar.Black);
                Cv2.ImShow($"{i}{nameof(thresh)}", gray);
               // Cv2.WaitKey();
             

                //    Cv2.Rectangle(gray, rect, Scalar.Red, 2);
                //    var roi = thresh[rect.Y, rect.Y + rect.Height, rect.X, rect.X + rect.Width];
                //    Cv2.BitwiseNot(roi, roi);
                //    Cv2.MedianBlur(roi, roi, 5);
                //    Cv2.Resize(roi, roi, new OpenCvSharp.Size(50, 50));
                //    mats.Add(roi);
                //    string licensePlate = String.Empty;
                //    var configVars = new KeyValuePair<string, string>("tessedit_char_whitelist",
                //        "ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890");
                //    for (int index = 0; index < mats.Count; index++)
                //    {
                //        var bmp = new Bitmap(mats[index].ToMemoryStream());
                //        using (var stream = TesseractSharp.Tesseract.ImageToTxt(bmp,
                //                   languages: new[] { Language.English, Language.Russian },
                //                   oem: OcrEngineMode.OemLstmOnly, psm: PageSegMode.PsmSingleLine,
                //                   configVars: new[] { configVars }))
                //        {
                //            using (var sr = new StreamReader(stream))
                //            {
                //                licensePlate += sr.ReadLine();
                //            }
                //        }
                //    }
                //}
                //Cv2.DrawContours(gray, contours, -1, Scalar.Yellow);
                
                
                //Cv2.ImShow($"{i}{nameof(thresh)}", gray);
            }
            Cv2.WaitKey();
            
            
        }
        [TestMethod]
        public void FindNumberPlate()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Check\";
            const string outputPath = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\";
            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }
            const string modelPath = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\haarcascade_russian_plate_number.xml";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            var classifier = new CascadeClassifier(modelPath);
            foreach (var file in files)
            {
                Mat image = new();
                Rect[]? rects = null;
                var min = -14.0;
                var max = Math.Abs(min);
                var rotation = new RotateBicubic(min) { KeepSize = true };
                while (min < max)
                {
                    using (var result = rotation.Apply(new Bitmap(file)) ?? throw new ArgumentNullException("rotation.Apply(new Bitmap(files[i]))"))
                        image = result.ToMat();
                    rects =  classifier.DetectMultiScale(image);
                    if (rects.Length == 0)
                        rotation.Angle = min += 0.2;
                    else break;
                }
                Debug.Assert(rects != null, nameof(rects) + " != null");
                //Cv2.Rectangle(image, rects[0], new Scalar(255, 170, 28));
            //    Cv2.ImShow(nameof(image), image);
                image = image[new Rect(rects[0].X, rects[0].Y, rects[0].Width, rects[0].Height)];
              //  image = image.Resize(new Size(102, 34));
                image.SaveImage(Path.Combine(outputPath, Path.GetFileName(file)));
               
              //  Cv2.WaitKey();
            }
        }
        [TestMethod]
        public void ResaveNumberPlate()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            foreach (var t in files)
            {
                Bitmap? bitmap = null;
                using (var fs = File.Open(t, FileMode.Open)) bitmap = new Bitmap(fs);
                if (File.Exists(t)) File.Delete(t);
                //GaussianSharpen gaussian = new();
                //bitmap = gaussian.Apply(bitmap);
                var checker = new DocumentSkewChecker();
                var angle = checker.GetSkewAngle(Grayscale.CommonAlgorithms.BT709.Apply(bitmap));
                if (!(Math.Abs(Math.Abs(angle) - 90) < 0.1))
                {
                    var rotation = new RotateBicubic(-angle) { KeepSize = true };
                    bitmap = rotation.Apply(bitmap);
                }
                bitmap.Save(t);
            }
        }
        
    }
}