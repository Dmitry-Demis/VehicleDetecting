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
using tessnet2;
using Tesseract = tessnet2.Tesseract;
using System.Linq;
using System.Runtime.InteropServices;
using AForge.Math;

namespace TestData
{
    public class LockBitmap
    {
        Bitmap source = null;
        IntPtr Iptr = IntPtr.Zero;
        BitmapData bitmapData = null;

        public byte[] Pixels { get; set; }
        public int Depth { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }

        public LockBitmap(Bitmap source)
        {
            this.source = source;
        }

        /// <summary>
        /// Lock bitmap data
        /// </summary>
        public void LockBits()
        {
            try
            {
                // Get width and height of bitmap
                Width = source.Width;
                Height = source.Height;

                // get total locked pixels count
                int PixelCount = Width * Height;

                // Create rectangle to lock
                Rectangle rect = new Rectangle(0, 0, Width, Height);

                // get source bitmap pixel format size
                Depth = System.Drawing.Bitmap.GetPixelFormatSize(source.PixelFormat);

                // Check if bpp (Bits Per Pixel) is 8, 24, or 32
                if (Depth != 8 && Depth != 24 && Depth != 32)
                {
                    throw new ArgumentException("Only 8, 24 and 32 bpp images are supported.");
                }

                // Lock bitmap and return bitmap data
                bitmapData = source.LockBits(rect, ImageLockMode.ReadWrite,
                                             source.PixelFormat);

                // create byte array to copy pixel values
                int step = Depth / 8;
                Pixels = new byte[PixelCount * step];
                Iptr = bitmapData.Scan0;

                // Copy data from pointer to array
                Marshal.Copy(Iptr, Pixels, 0, Pixels.Length);
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        /// <summary>
        /// Unlock bitmap data
        /// </summary>
        public void UnlockBits()
        {
            try
            {
                // Copy data from byte array to pointer
                Marshal.Copy(Pixels, 0, Iptr, Pixels.Length);

                // Unlock bitmap data
                source.UnlockBits(bitmapData);
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        /// <summary>
        /// Get the color of the specified pixel
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public Color GetPixel(int x, int y)
        {
            Color clr = Color.Empty;

            // Get color components count
            int cCount = Depth / 8;

            // Get start index of the specified pixel
            int i = ((y * Width) + x) * cCount;

            if (i > Pixels.Length - cCount)
                throw new IndexOutOfRangeException();

            if (Depth == 32) // For 32 bpp get Red, Green, Blue and Alpha
            {
                byte b = Pixels[i];
                byte g = Pixels[i + 1];
                byte r = Pixels[i + 2];
                byte a = Pixels[i + 3]; // a
                clr = Color.FromArgb(a, r, g, b);
            }
            if (Depth == 24) // For 24 bpp get Red, Green and Blue
            {
                byte b = Pixels[i];
                byte g = Pixels[i + 1];
                byte r = Pixels[i + 2];
                clr = Color.FromArgb(r, g, b);
            }
            if (Depth == 8)
            // For 8 bpp get color value (Red, Green and Blue values are the same)
            {
                byte c = Pixels[i];
                clr = Color.FromArgb(c, c, c);
            }
            return clr;
        }

        /// <summary>
        /// Set the color of the specified pixel
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="color"></param>
        public void SetPixel(int x, int y, Color color)
        {
            // Get color components count
            int cCount = Depth / 8;

            // Get start index of the specified pixel
            int i = ((y * Width) + x) * cCount;

            if (Depth == 32) // For 32 bpp set Red, Green, Blue and Alpha
            {
                Pixels[i] = color.B;
                Pixels[i + 1] = color.G;
                Pixels[i + 2] = color.R;
                Pixels[i + 3] = color.A;
            }
            if (Depth == 24) // For 24 bpp set Red, Green and Blue
            {
                Pixels[i] = color.B;
                Pixels[i + 1] = color.G;
                Pixels[i + 2] = color.R;
            }
            if (Depth == 8)
            // For 8 bpp set color value (Red, Green and Blue values are the same)
            {
                Pixels[i] = color.B;
            }
        }
    }
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void Resize()
        {
            var s = new Mat(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-4-crop.png");
           s = s.Resize(new Size(), 5, 5, InterpolationFlags.Cubic);
            var bitmap = s.ToBitmap();
            bitmap.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\res.png", ImageFormat.Png);
        }
        [TestMethod]
        public void Un()
        {
            var s = new Bitmap(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\otsu.png");
            Mat image = new( 120, s.Width, MatType.CV_8UC1, Scalar.White);
            
            int[] r = new int[s.Width];
            r.Select(r => 10 * r);
            for (int i = 0; i < s.Width; i++)
            {
                for (int j = 0; j < s.Height; j++)
                {
                    var c = s.GetPixel(i, j);
                    if ( c == Color.FromArgb(255,255,255,255))
                    {
                        r[i] += 1;
                    }
                }
                Cv2.Line(image, new Point(i, 0), new Point(i, r[i]), Scalar.Black, 2);
                image = image.Resize(new Size(s.Width, 350));
                Cv2.ImShow("sa", image);
            }
            
            Cv2.WaitKey();

        }
        [TestMethod]
        public void EmptyOrIsVehicle()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Train\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
          //  VehicleAvailability vehicle = new();
           // vehicle.LoadModel($@"..\..\..\..\..\Helper\Models\carEmpty.h5");
            foreach (var file in files)
            {
          //      var @class = int.Parse(Path.GetFileName(file)[0].ToString());
          //      var result = vehicle.RecognizeVehicle(new Bitmap(file));
          //      Assert.AreEqual(result, @class == 1);
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
                var coeff = 7;
                Cv2.Resize(gray, gray, new Size(), fx: coeff, fy: coeff, InterpolationFlags.Cubic);
                Mat thresh = new();
                Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.BinaryInv);
                Mat rect_kern = new();
                rect_kern =  Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5,5)); 
                Mat dilation = new();
                Cv2.Dilate(thresh, dilation, rect_kern);
              Cv2.FindContours(dilation, out var contours, out var hierarchy, RetrievalModes.Tree,
                ContourApproximationModes.ApproxSimple);
              Cv2.ImShow($"{i}{nameof(thresh)}", dilation);
                Array.Sort(contours, ((points, points1) =>
                {
                    var l = Cv2.BoundingRect(points);
                    var r = Cv2.BoundingRect(points1);
                    return (l.X > r.X) ? 1 : -1;
                }));
                List<Point[]> mats = new ();
                //Cv2.DrawContours(gray, new List<Point[]> { contours[0] }, -1, Scalar.Black);
                Cv2.ImShow($"{i}{nameof(thresh)}", dilation);
                //Cv2.WaitKey();
                for (var index1 = 0; index1 < contours.Length; index1++)
                {
                    var contour = contours[index1];
                    var rect = Cv2.BoundingRect(contour);
                    if ((1.0 * thresh.Height) / rect.Height > 6)
                    {
                        continue;
                    }
                    var ratio = (1.0 * rect.Height) / rect.Width;
                    if (ratio < 0.9)
                    {
                        continue;
                    }
                    var area = Cv2.ContourArea(contour);
                    if (area < 1300)
                    {
                        continue;
                    }

                    if (Cv2.BoundingRect(contours[index1]).X + Cv2.BoundingRect(contours[index1]).Width < Cv2.BoundingRect(contours[index1-1]).X + Cv2.BoundingRect(contours[index1-1]).Width)
                    {
                        continue;
                    }
                    mats.Add(contour);
                }
                string licensePlate = String.Empty; 
                var configVars = new KeyValuePair<string, string>("tessedit_char_whitelist",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890");
                var ocr = new Tesseract();
                ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890¿¡¬√ƒ≈®∆«»… ÀÃÕŒœ–—“”‘’÷◊ÿŸ⁄€‹›ﬁﬂ");
                ocr.Init(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\rus.traineddata", "rus", false);
                for (int j = 0; j < mats.Count; j++)
                {
                    var rect = Cv2.BoundingRect(mats[j]);
                    Cv2.Rectangle(gray, rect, Scalar.Black);
                    var roi = thresh[rect.Y, rect.Y + rect.Height, rect.X, rect.X + rect.Width];
                    Cv2.BitwiseNot(roi, roi);
                    Cv2.MedianBlur(roi, roi, 5);
                    Cv2.Resize(roi, roi, new OpenCvSharp.Size(50, 50));
                    Cv2.ImShow(nameof(roi), roi);
                 //  Cv2.WaitKey();
                    var bmp = roi.ToBitmap();


                    List<tessnet2.Word> result = ocr.DoOCR(bmp, Rectangle.Empty);
                    //using (var stream = TesseractSharp.Tesseract.ImageToTxt(bmp,
                    //           languages: new[] { Language.Russian },
                    //           oem: OcrEngineMode.OemLstmOnly, psm: PageSegMode.PsmSingleLine,
                    //           configVars: new[] { configVars }))
                    //{
                    //    using (var sr = new StreamReader(stream))
                    //    {
                    //        var str = sr.ReadLine();
                    //        if (licensePlate.Length is > 0 and < 5) 
                    //        {
                    //            if (str == "O")
                    //            {
                    //                str = "0";
                    //            }
                    //        }
                    //        else if (licensePlate.Length == 6)
                    //        {
                    //            if (str == "J")
                    //            {
                    //                str = "7";
                    //            }
                    //        }

                    //        if (str == "Y")
                    //        {
                    //            str = "”";
                    //        }

                    //        licensePlate += str;
                    //    }
                    //}
                }
                Cv2.ImShow($"{i}{nameof(thresh)}", gray);
                Cv2.WaitKey();


                //    Cv2.Rectangle(gray, rect, Scalar.Red, 2);
             
                //   
                //    mats.Add(roi);
                //    
                //    for (int index = 0; index < mats.Count; index++)
                //    {
                //        
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
              image = image.Resize(new Size(85, 30));
                image.SaveImage(Path.Combine(outputPath, Path.GetFileName(file)));
               
              //  Cv2.WaitKey();
            }
        }

        public Bitmap FindLicensePlate(Bitmap image, string modelPath)
        {
            Rect[]? rects = null;
            var classifier = new CascadeClassifier(modelPath);
            var min = -14.0;
            var max = Math.Abs(min);
            var rotation = new RotateBicubic(min) { KeepSize = true };
            Mat to = new();
            while (min < max)
            {
                using (var result = rotation.Apply(new Bitmap(image)) ?? throw new ArgumentNullException())
                    to = result.ToMat();
                rects = classifier.DetectMultiScale(to);
                if (rects.Length == 0)
                    rotation.Angle = min += 0.2;
                else break;
            }
            if (rects != null) to = to[new Rect(rects[0].X, rects[0].Y, rects[0].Width, rects[0].Height)];
            to = to.Resize(new Size(85, 30));
            return to.ToBitmap();
        }

        public Bitmap HoughTransformation(Bitmap image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            var checker = new DocumentSkewChecker();
            if (image.PixelFormat != PixelFormat.Format8bppIndexed)
                image = Grayscale.CommonAlgorithms.BT709.Apply(image);
            var angle = checker.GetSkewAngle(image);
            if (!(Math.Abs(Math.Abs(angle) - 90) < 0.1))
            {
                var rotation = new RotateBicubic(-angle) { KeepSize = true };
                image = rotation.Apply(image);
            }
            return image;
        }

        public Bitmap LaplacianFilter(Bitmap image, int[,] filter)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (filter == null) throw new ArgumentNullException(nameof(filter));
            if (image.PixelFormat != PixelFormat.Format8bppIndexed)
                image = Grayscale.CommonAlgorithms.BT709.Apply(image);
            var convolution = new Convolution(filter);
            image = convolution.Apply(image);
            return image;
        }

        [TestMethod]
        public void Test()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-4.png";
            var image = new Bitmap(path);
            var filter = new[,] { { 1, 1, 1 }, { 1, -8, 1 }, { 1, 1, 1 } };
            image = LaplacianFilter(image, filter);
            image.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\n.png");
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

        [TestMethod]
        public void PrepareSymbols()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\";
            var files = Directory.GetFiles(path, "*.png") ?? throw new ArgumentNullException("Directory.GetFiles(path, \"*.png\")");
            var count = 0;
            foreach (var file in files)
            {
                Mat mat = new Mat(file);
                Mat gray = mat.CvtColor(ColorConversionCodes.BGR2GRAY);
                Mat thresh = new();
                Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.Otsu);
                Cv2.FindContours(thresh, out var contours, out var hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
                for (int j = 0; j < contours.Length; j++)
                {
                    var r = Cv2.BoundingRect(contours[j]);
                    var area = Cv2.ContourArea(contours[j]);
                    if (r.Height/(r.Width * 1.0 )> 1.1 && area > 1200)
                    {
                        Mat tmp = gray[new Rect(r.X - 10, r.Y - 10, r.Width + 20, r.Height + 20)];
                        tmp = tmp.Resize(new Size(72, 102));
                        tmp.SaveImage(@$"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\{count++}.png");
                        Cv2.Rectangle(mat, new Rect(r.X - 10, r.Y - 10, r.Width + 20, r.Height + 20), Scalar.Red, 2);
                        
                    }
                    
                } // Cv2.DrawContours(mat, contours, -1, Scalar.Red, 2);
            //    Cv2.ImShow("s", mat);
           //     Cv2.WaitKey();
            }
        }

        public List<Mat> FindContoursForDetecting(Bitmap bitmap, int resizeCoeff)
        {
            var image = bitmap.ToMat();
            image = image.Resize(new Size(), resizeCoeff, resizeCoeff, InterpolationFlags.Cubic);
            return null;
        }
        [TestMethod]
        public void Crop()
        {
            const string subpath = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-5";
            const string ext = ".png";
            int coeff = 10;
            Mat image = new Mat(subpath + ext, ImreadModes.Grayscale);
            Mat imgCropped = image.Clone();
           // imgCropped = imgCropped.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
          //  imgCropped = imgCropped.GaussianBlur(new Size(5, 5), 0);
           // image = image.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
          image = image.GaussianBlur(new Size(5, 5), 0);
          image = image.Threshold(0, 255, ThresholdTypes.Otsu);
            /*Cv2.ImShow("i", image);
            Cv2.WaitKey();*/
            Cv2.FindContours(image, out var contours, out var hierarchy, RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);
            Array.Sort(contours, (points, points1) =>
            {
                var left = Cv2.ContourArea(points);
                var right = Cv2.ContourArea(points1);
                return !(left < right) ? -1 : 1;
            });
            var c = 0;
            var rect = Cv2.BoundingRect(contours[0]);
            Rect r = new Rect(rect.X, rect.Y+c, rect.Width, rect.Height-c);
            imgCropped = imgCropped[r];
            Bitmap bitmap = imgCropped.ToBitmap();
            bitmap.Save(subpath + "-crop" + ext);
            /*Cv2.ImShow("i", imgCropped);
            
            Cv2.WaitKey();*/

        }
        [TestMethod]
        public void FindContours()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\0.png";
            const string path2 = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-4-crop.png";
            int coeff = 5;
            Mat img = new Mat(path2);
            img = img.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
            Mat image = new Mat(path, ImreadModes.Grayscale);
            
            image = image.Resize(new Size(), coeff, coeff, InterpolationFlags.Cubic);
            //Cv2.ImShow("i", image);
            //Cv2.WaitKey();
            image = image.GaussianBlur(new Size(3, 3), 0);
            image = image.Threshold(0, 255, ThresholdTypes.Otsu);
           
           // var rect_kern = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
          //  Mat dilation = new();
           //// Cv2.Dilate(image, dilation, rect_kern);
           //Cv2.ImShow("i", image);
           //Cv2.WaitKey();
            Cv2.FindContours(image, out var contours, out var hierarchy, RetrievalModes.Tree,
                ContourApproximationModes.ApproxSimple);
      Cv2.DrawContours(img, contours, -1, Scalar.Cyan);
            //for (int i = 0; i < contours.Length; i++)
            //{
            //    Cv2.Rectangle(img, Cv2.BoundingRect(contours[i]), Scalar.Cyan);
            //  }
            //Cv2.ImShow("i", img);
            //Cv2.WaitKey();
            //var rects = Cv2.BoundingRect(contours[0]);
            //var max = Cv2.ContourArea(contours[0]);
            //for (int i = 1; i < contours.Length; i++)
            //{
            //    if (Cv2.ContourArea(contours[i]) > max)
            //    {
            //        rects = Cv2.BoundingRect(contours[i]);
            //        max = Cv2.ContourArea(contours[i]);
            //    }
            //}

            //var k = 10;
            //img = img[rects.Y + k, rects.Y + rects.Width - k, rects.X + k, rects.X + rects.Height - k];
            //Cv2.ImShow("i", img);
            //Cv2.WaitKey();
            Array.Sort(contours, (points, points1) =>
            {
                var left = Cv2.BoundingRect(points);
                var right = Cv2.BoundingRect(points1);
                return left.X < right.X? -1 : 1;
            });

            List<Point[]> ct = new ();
            for (int i = 0; i < contours.Length; i++)
            {
                var rect = Cv2.BoundingRect(contours[i]);
                 var rect2 = (i != 0 && ct.Count!=0) ? Cv2.BoundingRect(ct[^1]) : Rect.Empty;
                 var area = Cv2.ContourArea(contours[i]);
                 Mat res = new();
                var s = Cv2.ApproxPolyDP(contours[i], 5, true);

                
                if (rect.Width is > 50 or <= 12)
                {
                    continue;
                }
                if (rect.Height <= 12)
                {
                    continue;
                }
                if (i != 0 && rect.X + rect.Width < rect2.X + rect2.Width && rect.Y + rect.Height < rect2.Y + rect2.Height)
                {
                    continue;
                }
                if (rect.Height * 1.0 / rect.Width is <= 0.8 or >= 3.0)
                {
                    continue;
                }

                /*if (ct.Count==8)
                {
                    break;
                }*/
                Cv2.Rectangle(img, Cv2.BoundingRect(contours[i]), Scalar.Red);
                ct.Add(contours[i]);
            }

            List<Point[]> n = new();
            for (int i = 1; i < ct.Count; i++)
            {
                var b = Cv2.BoundingRect(ct[i]);
                var bs = Cv2.BoundingRect(ct[i-1]);
                if (b.X-bs.X<10)
                {
                    n.Add(ct[i]);
                }
            }
            List<double> ar = new List<double>();
            List<(int, int, int, int, double, double)> frac = new();
            for (int i = 0; i < contours.Length; i++)
            {
              
                var rect = Cv2.BoundingRect(contours[i]);
                frac.Add((rect.X, rect.Y, rect.Width, rect.Height, rect.Height * 1.0 / rect.Width, Cv2.ContourArea(contours[i])));

            }

            for (int j = 0; j < ct.Count; j++)
            {
                var r = Cv2.BoundingRect(ct[j]);
                Mat tmp = image[new Rect(r.X, r.Y, r.Width, r.Height)];
                tmp = tmp.Resize(new Size(72, 102));
                Cv2.BitwiseNot(tmp, tmp);
                var res = tmp.ToBitmap();
                //tmp = tmp.CvtColor(ColorConversionCodes.BGR2GRAY);
                res.Save(@$"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Result\{j}.png", ImageFormat.Png);
               
            }
            // Cv2.DrawContours(img, contours, -1, Scalar.Red);
            Cv2.ImShow(nameof(img), img);
            Cv2.WaitKey();
            int a = 5;
            a = 7;
            Console.WriteLine();
        }
        [TestMethod]
        public void Filters()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-4-crop.png";

            {
                Bitmap? bitmap = new Bitmap(path);
             //bitmap = Grayscale.CommonAlgorithms.BT709.Apply(bitmap);
             //BrightnessCorrection b = new(50);
            // bitmap = b.Apply(bitmap);
                OtsuThreshold otsu = new OtsuThreshold();
                Sharpen sharpen = new ();


              //  var kernel1 = new[,] { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };
               var kernel2 = new[,] { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };
                //var kernel2 = new[,] { { 0, 1, 0 }, { 1, -5, 1 }, { 0, 1, 0 } };
             //   var kernel3 = new[,] { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
                var kernel4 = new[,] { { 1, 1, 1 }, { 1, -8, 1 }, { 1, 1, 1 } };
               // var kernel4 = new[,] { { 1, 1, 1 }, { 1, -9, 1 }, { 1, 1, 1 } };
                List<int[,]> list = new List<int[,]>() {    kernel4 };
                for (int i = 0; i < list.Count; i++)
                {
                    sharpen.Kernel = list[i];
                    var c = sharpen.Apply(bitmap);

                    c.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\" + $"{i}.png");
                }
                
              //  var k = otsu.Apply(bitmap);
              //  k.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\" + $"otsu.png");

            }
        }

        [TestMethod]
        public void Otsu()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\¡ÂÁ ËÏÂÌË-4-crop.png";
            Bitmap? bitmap = new Bitmap(path);
            //bitmap = Grayscale.CommonAlgorithms.BT709.Apply(bitmap);
            //GaussianSharpen s = new(20);
           // bitmap = s.Apply(bitmap);
            OtsuThreshold otsu = new OtsuThreshold();
            var r = otsu.ThresholdValue;
            var k = otsu.Apply(bitmap);
           
            k.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\" + $"otsu.png");
        }
        [TestMethod]
        public void Brightness()
        {
            const string path = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\1.png";
            Mat m = new Mat(path);
         //  m =  m.Resize(new Size(), 5, 5, InterpolationFlags.Cubic);
            var rect_kern = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
            Mat dilation = new();
            Cv2.Dilate(m, dilation, rect_kern);
            var k = dilation.ToBitmap();
            k.Save(@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Numbers\" + $"otsu.png");
        }

        [TestMethod]
        public void CreateDirectories()
        {
            var array = "0,1,2,3,4,5,6,7,8,9,A,B,E,K,M,H,O,P,C,T,Y,X";
            var split = array.Split(',');
            for (int i = 0; i < split.Length; i++)
            {
                Directory.CreateDirectory(
                    $@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\{split[i]}");
            }
        }

        [TestMethod]
        public void RenameEverything()
        {
            var array = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21";
            var split = array.Split(',');
            //for (int i = 0; i < split.Length; i++)
           for(int i = 0; i< split.Length; i++)
            {
                var files = Directory.GetFiles(
                    $@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\{split[i]}");
                for (int j = 0; j < files.Length; j++)
                {
                    Bitmap? bitmap = null;
                    using (var fs = File.Open(files[j], FileMode.Open)) bitmap = new Bitmap(fs);
                    if (File.Exists(files[j])) File.Delete(files[j]);
                  //  bitmap = Grayscale.CommonAlgorithms.BT709.Apply(bitmap);
                    bitmap.Save(
                        $@"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\{split[i]}\{split[i]}_{j}_image.png",
                        ImageFormat.Png);
                }

            };
        }

        [TestMethod]
        public void TrainSymbols()
        {
            var trainDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Train\";
            var testDir = @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Images\Symbols\symbols\Test\";
            //var testDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Test\test\";
            //var valDir = @"D:\HEI\BLOCK 4C\Diploma\DetectingVehicle\NeuralNet\Images\Val\val\";
            //CNN_ForCarDetecting((440, 100, 3), trainDir, testDir, valDir, 20, 20, 400, 200, 200, classMode: "binary");
         //   Numbers numbers = new Numbers();
         //   numbers.Train((72,102,3), trainDir, testDir, null, 60, 1, 2, 0, 2,
         //       @"D:\HEI\BLOCK 4C\Diploma\VehicleDetecting\Helper\Models\", 22);
        }
    }
}