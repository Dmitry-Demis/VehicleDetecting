using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Helper;
using Microsoft.Win32;
using Brushes = System.Windows.Media.Brushes;
using Image = System.Drawing.Image;
using Path = System.IO.Path;
using Rectangle = System.Drawing.Rectangle;

namespace VehicleDetecting
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Keras.Keras.DisablePySysConsoleLog = true;
            DataContext = this;
        }
        private VehicleAvailability? _model;
        private Bitmap? _image;
        #region readonly LoadedImage : bool - Проверка на добавленное фото
        /// <summary>Проверка на добавленное фото</summary>
        public static readonly DependencyProperty LoadedImageProperty =
            DependencyProperty.Register(
                nameof(LoadedImage),
                typeof(bool),
                typeof(MainWindow),
                new PropertyMetadata(default(bool)));

        /// <summary>Проверка на добавленное фото</summary>
        [Description("Проверка на добавленное фото")]
        public bool LoadedImage
        {
            get => (bool)GetValue(LoadedImageProperty);
            set => SetValue(LoadedImageProperty, value);
        }
        #endregion
        private void RecognizeButton_Click(object sender, RoutedEventArgs e)
        {
            string? pathModel = null;
            if (modelsPathCheckBox.IsChecked == false)
            {
                var op = new OpenFileDialog
                {
                    Title = "Выбрать файл модели для распознавания",
                    Filter = "Model|*.h5"
                };
                if (op.ShowDialog() == true) pathModel = op.FileName;
            }
            else
            {
                var path = Path.GetFullPath(@"..\..\..\..\Helper\Models\carEmpty.h5");
                if (!File.Exists(path))
                    throw new FileLoadException("The model for vehicle detecting is absent");
                pathModel = path;
            }
          //  if (pathModel != null) _model = new VehicleAvailability(pathModel);
            //else
            {
                throw new NullReferenceException("Path model is absent");
            }
            resultTxtBlock.Foreground = Brushes.DeepSkyBlue;
            resultTxtBlock.Text = "Ожидайте...";
          //  var result = _model.RecognizeVehicle(_image);
           // if (!result)
            {
                resultTxtBlock.Foreground = Brushes.Red;
                resultTxtBlock.Text = "ТС отсутствует";
            }
            //new Rectangle(400, 240, 440, 100)
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            var op = new OpenFileDialog
            {
                Title = "Выбрать картинку",
                Filter = "All supported graphics|*.jpg;*.jpeg;*.png|" +
                         "JPEG (*.jpg;*.jpeg)|*.jpg;*.jpeg|" +
                         "Portable Network Graphic (*.png)|*.png"
            };
            if (op.ShowDialog() == true)
            {
                loadedImage.Source = new BitmapImage(new Uri(op.FileName));
                _image = new Bitmap(op.FileName);
                LoadedImage = true;
            }
            else
                LoadedImage = false;
        }
    }
}
