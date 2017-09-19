using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
namespace HDISED_GPU
{
    
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // CudafyModes.Target = eGPUType.Cuda; // Jak chcesz obliczac na CUDA
            CudafyModes.Target = eGPUType.OpenCL; // Jak chcesz obliczac na OpenCL

            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;
            try
            {
                int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
                if (deviceCount == 0)
                {
                    Console.WriteLine("No suitable {0} devices found.", CudafyModes.Target);
                    throw new Exception("No suitable devices found.");
                }
                GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
                Console.WriteLine("Running examples using {0}", gpu.GetDeviceProperties(false).Name);


                SQL.Execute();    // glowna funkcja
                //SQL.Test();         // testowa funkcja               
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
