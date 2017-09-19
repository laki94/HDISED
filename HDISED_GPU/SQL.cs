using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace HDISED_GPU
{
    public class SQL
    {
        public const int N = 10;

        public static SqlConnection connectToServer(String server, String database, String user, String passwd)
        {
            SqlConnection connection = new SqlConnection();
            connection.ConnectionString = "Server=" + server + "; Database=" + database + "; User Id=" + user + "; password=" + passwd;
            connection.Open();
            return connection;
        }

        public static void Execute()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

           
        }

        public static void Test()
        {
            int min = 0;
            int max = 0;
            int avg = 0;

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            var archivesConnection = connectToServer("localhost", "Archives", "sa", "root");
            var selectAll = new SqlCommand("SELECT TOP (" + N + ") FuelVolume FROM RefuelStream");
            selectAll.Connection = archivesConnection;
            var reader = selectAll.ExecuteReader();

            float[] receivedValues = new float[N];
            float[] calculatedValues = new float[3];

            float[] recValGpu = gpu.Allocate<float>(receivedValues);
            float[] calValGpu = gpu.Allocate<float>(calculatedValues);

            for (int i = 0; i < N; i++)
                receivedValues[i] = 0;
            for (int i = 0; i < 3; i++)
                calculatedValues[i] = 0;

            for (int i = 0; reader.Read(); i++)
            {
                Object value = reader.GetValue(0);
                receivedValues[i] = float.Parse(value.ToString());
            }
                reader.Close();

            gpu.CopyToDevice(receivedValues, recValGpu);
            gpu.CopyToDevice(calculatedValues, calValGpu);

            gpu.Launch().calculateAggregation(recValGpu, calValGpu); // Launch() bo dzialamy na 1 watku, mozna dac Launch(4, 1) wtedy jedziemy na 4 watkach ale trzeba blokowac zasoby czego nie ogarnalem.

            gpu.CopyFromDevice(calValGpu, calculatedValues);

            min = (int)Math.Round(calculatedValues[0]); // nie ogarnalem tez wkladania floatow do tabeli dlatego takie czary
            max = (int)Math.Round(calculatedValues[1]);
            avg = (int)Math.Round(calculatedValues[2]);

            var dataConnection = connectToServer("localhost", "Data", "sa", "root");
            var insertData = new SqlCommand("INSERT INTO TankMeasures(FuelLevel, FuelVolume, FuelTemperature) VALUES (" + min + ", " + max + ", " + avg + ")");
            insertData.Connection = dataConnection;
            insertData.ExecuteNonQuery();

            archivesConnection.Close();
            dataConnection.Close();
        }

        [Cudafy]
        public static void calculateAggregation(GThread thread, float[] recValGPU, float[] calValGPU)
        {
            int tid = thread.blockIdx.x;
            while (tid < N)
            {
                if ((calValGPU[0] == 0) || (recValGPU[tid] < calValGPU[0])) // MIN
                    calValGPU[0] = recValGPU[tid];

                if ((calValGPU[1] == 0) || (recValGPU[tid] > calValGPU[1])) // MAX
                    calValGPU[1] = recValGPU[tid];

                calValGPU[2] += recValGPU[tid];

                if (tid == N-1)                                     // AVG
                    calValGPU[2] /= N;

                tid += thread.gridDim.x;
            }
        }
    }
}
