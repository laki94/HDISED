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
        public const int WAITTIME = 30;

        public static int lastTimeStamp = 100;
        public static SqlConnection archivesConnection;
        public static SqlConnection dataConnection;
        public static SqlConnection resultsConnection;

        enum TankID
        {
            first = 1, 
            second = 2, 
            third = 3,
            fourth = 4
        }

        public static SqlConnection connectToServer(String server, String database, String user, String passwd)
        {
            SqlConnection connection = new SqlConnection();
            connection.ConnectionString = "Server=" + server + "; Database=" + database + "; User Id=" + user + "; password=" + passwd;
            connection.Open();
            return connection;
        }

        public static SqlCommand getParameterCommand(string table, string parameter, int timeStamp)
        {
            SqlCommand query = new SqlCommand();
            float[] result = new float[Enum.GetNames(typeof(TankID)).Length];
            if (dataConnection.State != System.Data.ConnectionState.Open)
            {
                dataConnection.Close();
                dataConnection.Open();
            }
            query.Connection = dataConnection;
            query.CommandText = "Select " + parameter + " from " + table + " where TimeId = " + timeStamp;
            return query;
        }

        public static float[] getFuelLevel(SqlCommand _query)
        {
            var tmpQuery = _query;
            string orgCommText = _query.CommandText;
            var reader = tmpQuery.ExecuteReader();
            Array enumItems = Enum.GetValues(typeof(TankID));

            float[] result = new float[enumItems.Length];
            foreach(int id in enumItems)
            {
                tmpQuery.CommandText += " and TankId = " + id;
                if (reader.Read())
                    result[id - 1] = Convert.ToSingle(reader.GetValue(0));
                else
                    throw new Exception("Nie mozna odnalezc danych dla podanych parametrow");
                tmpQuery.CommandText = orgCommText;
            }
            reader.Close();
            return result;
        }

        public static void Execute()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            dataConnection = connectToServer("localhost", "Data", "sa", "root");

            float[] previousMeasures = getFuelLevel(getParameterCommand("TankMeasures", "FuelVolume", lastTimeStamp));
            float[] actualMeasures = getFuelLevel(getParameterCommand("TankMeasures", "FuelVolume",  lastTimeStamp + WAITTIME));
            lastTimeStamp += WAITTIME;
            float[] previousMeasuresGPU = gpu.Allocate<float>(previousMeasures);
            float[] actualMeasuresGPU = gpu.Allocate<float>(actualMeasures);
            gpu.CopyToDevice(previousMeasures, previousMeasuresGPU);
            gpu.CopyToDevice(actualMeasures, actualMeasuresGPU);

            gpu.Launch().calculateFuelVolume(previousMeasuresGPU, actualMeasuresGPU);

           

            for (int i = 0; i < previousMeasures.Length; i++)
            {
                Console.WriteLine("PREV: " + previousMeasures[i]);
                Console.WriteLine("ACT: " + actualMeasures[i]);
            }
            gpu.CopyFromDevice(previousMeasuresGPU, previousMeasures);

            for (int i=0;i<previousMeasures.Length;i++)
                Console.WriteLine("OBLICZONE: " + previousMeasures[i]);
                
           
        }

        public static void Test()
        {
            int min = 0;
            int max = 0;
            int avg = 0;

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            archivesConnection = connectToServer("localhost", "Archives", "sa", "root");
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

            dataConnection = connectToServer("localhost", "Data", "sa", "root");
            var insertData = new SqlCommand("INSERT INTO TankMeasures(FuelLevel, FuelVolume, FuelTemperature) VALUES (" + min + ", " + max + ", " + avg + ")");
            insertData.Connection = dataConnection;
            insertData.ExecuteNonQuery();

            archivesConnection.Close();
            dataConnection.Close();
        }

        [Cudafy]
        public static void calculateFuelVolume(GThread thread, float[] prevMeasures, float[] actMeasures)
        {
            int tid = thread.blockIdx.x;
            while (tid < prevMeasures.Length) //(tid < N)
            {
                if ((prevMeasures[tid] -= actMeasures[tid]) != 0)
                    prevMeasures[tid] /= WAITTIME;

                tid += thread.gridDim.x;
            }
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
