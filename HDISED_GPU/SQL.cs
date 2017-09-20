using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Globalization;

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

        public static CudafyModule km;
        public static GPGPU gpu;

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

        public static string[] executeForEveryTank(SqlCommand _query)
        {
            var tmpQuery = _query;
            string orgCommText = _query.CommandText;
            var reader = tmpQuery.ExecuteReader();
            Array enumItems = Enum.GetValues(typeof(TankID));

            string[] result = new string[enumItems.Length];
            foreach(int id in enumItems)
            {
                tmpQuery.CommandText += " and TankId = " + id;
                if (reader.Read())
                    result[id - 1] = reader.GetValue(0).ToString();//Convert.ToSingle(reader.GetValue(0));
                else
                    throw new Exception("Nie mozna odnalezc danych dla podanych parametrow");
                tmpQuery.CommandText = orgCommText;
            }
            reader.Close();
            return result;
        }

        private static int[][] getPrevAndActValuesInt(SqlConnection dataConnection, string parameter)
        {
            int[][] result = new int[2][];
            string[] tmpPrev = executeForEveryTank(getParameterCommand("TankMeasures", parameter, lastTimeStamp));
            string[] tmpAct = executeForEveryTank(getParameterCommand("TankMeasures", parameter, lastTimeStamp + WAITTIME));
            int[] previousMeasures = new int[Enum.GetValues(typeof(TankID)).Length];
            int[] actualMeasures = new int[Enum.GetValues(typeof(TankID)).Length];

            for (int i = 0; i < previousMeasures.Length; i++)
            {
                previousMeasures[i] = Int32.Parse(tmpPrev[i]);
                actualMeasures[i] = Int32.Parse(tmpAct[i]);
            }

            result[0] = previousMeasures;
            result[1] = actualMeasures;
            return result;
        }

        private static float[][] getPrevAndActValuesFloat(SqlConnection dataConnection, string parameter)
        {
            float[][] result = new float[2][];
            string[] tmpPrev = executeForEveryTank(getParameterCommand("TankMeasures", parameter, lastTimeStamp));
            string[] tmpAct = executeForEveryTank(getParameterCommand("TankMeasures", parameter, lastTimeStamp + WAITTIME));
            float[] previousMeasures = new float[Enum.GetValues(typeof(TankID)).Length];
            float[] actualMeasures = new float[Enum.GetValues(typeof(TankID)).Length];

            for (int i = 0; i < previousMeasures.Length; i++)
            {
                previousMeasures[i] = float.Parse(tmpPrev[i]);
                actualMeasures[i] = float.Parse(tmpAct[i]);
            }

            result[0] = previousMeasures;
            result[1] = actualMeasures;
            return result;
        }



        public static void prepareGPU()
        {
            km = CudafyTranslator.Cudafy(); 
            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);
        }

        public static float[] prepareAndCalculateFloatData(float[] prevMeasures, float[] actMeasures)
        {
            float[] previousMeasuresGPU = gpu.Allocate<float>(prevMeasures);
            float[] actualMeasuresGPU = gpu.Allocate<float>(actMeasures);
            gpu.CopyToDevice(prevMeasures, previousMeasuresGPU);
            gpu.CopyToDevice(actMeasures, actualMeasuresGPU);

            gpu.Launch(prevMeasures.Length, 1).calculateDataWithCudafy(previousMeasuresGPU, actualMeasuresGPU);
            gpu.CopyFromDevice(previousMeasuresGPU, prevMeasures);
            gpu.FreeAll();
            return prevMeasures;
        }

        public static int[] prepareAndCalculateIntData(int[] prevMeasures, int[] actMeasures)
        {
            int[] previousMeasuresGPU = gpu.Allocate<int>(prevMeasures);
            int[] actualMeasuresGPU = gpu.Allocate<int>(actMeasures);
            gpu.CopyToDevice(prevMeasures, previousMeasuresGPU);
            gpu.CopyToDevice(actMeasures, actualMeasuresGPU);

            gpu.Launch(prevMeasures.Length, 1).calculateDataWithCudafy(previousMeasuresGPU, actualMeasuresGPU);
            gpu.CopyFromDevice(previousMeasuresGPU, prevMeasures);
            gpu.FreeAll();
            return prevMeasures;
        }

        public static void sendResultsToDatabase(string table, int[] fuelLevel = null, float[] fuelVolume = null, int[] fuelTemperature = null, int[] waterLevel = null, float[] waterVolume = null)
        {
            var tmpQuery = "INSERT INTO " + table + " VALUES (";
            var calcConnection = connectToServer("localhost", "Calculations", "sa", "root");
            SqlCommand insertData = new SqlCommand(tmpQuery);
            NumberFormatInfo nfi = new NumberFormatInfo();
            nfi.NumberDecimalSeparator = ".";

            Array enumItems = Enum.GetValues(typeof(TankID));
            foreach (int id in enumItems)
            {
                insertData.CommandText += lastTimeStamp + ", " + id;
                if (fuelLevel != null)
                    insertData.CommandText += ", " + fuelLevel[id - 1]; 
                else
                    insertData.CommandText += ", " + 0; 

                if (fuelVolume != null)
                    insertData.CommandText += ", " + fuelVolume[id - 1].ToString(nfi);
                else
                    insertData.CommandText += ", " + 0; 

                if (fuelTemperature != null)
                    insertData.CommandText += ", " + fuelTemperature[id - 1];
                else
                    insertData.CommandText += ", " + 0; 

                if (waterLevel != null)
                    insertData.CommandText += ", " + waterLevel[id - 1];
                else
                    insertData.CommandText += ", " + 0; 

                if (waterVolume != null)
                    insertData.CommandText += ", " + waterVolume[id - 1].ToString(nfi);
                else
                    insertData.CommandText += ", " + 0; 

                insertData.CommandText += ")";
                insertData.Connection = calcConnection;
                insertData.ExecuteNonQuery();
                insertData.CommandText = tmpQuery;   
            }
            calcConnection.Close();
        }

        public static void Execute()
        {
            const int PREVIOUS_MEASURES = 0;
            const int ACTUAL_MEASURES = 1;

            dataConnection = connectToServer("localhost", "Data", "sa", "root");
            prepareGPU();

            float[][] fuelVolumeMeasures = getPrevAndActValuesFloat(dataConnection, "FuelVolume");
            float[] resFuelVol = prepareAndCalculateFloatData(fuelVolumeMeasures[PREVIOUS_MEASURES], fuelVolumeMeasures[ACTUAL_MEASURES]);

            int[][] fuelLevelMeasures = getPrevAndActValuesInt(dataConnection, "FuelLevel");
            int[] resFuelLevel = prepareAndCalculateIntData(fuelLevelMeasures[PREVIOUS_MEASURES], fuelLevelMeasures[ACTUAL_MEASURES]);

            int[][] fuelTemperature = getPrevAndActValuesInt(dataConnection, "FuelTemperature");
            int[] resFuelTemperature = prepareAndCalculateIntData(fuelTemperature[PREVIOUS_MEASURES], fuelTemperature[ACTUAL_MEASURES]);

            int[][] waterLevel = getPrevAndActValuesInt(dataConnection, "WaterLevel");
            int[] resWaterLevel = prepareAndCalculateIntData(waterLevel[PREVIOUS_MEASURES], waterLevel[ACTUAL_MEASURES]);

            float[][] waterVolume = getPrevAndActValuesFloat(dataConnection, "WaterVolume");
            float[] resWaterVolume = prepareAndCalculateFloatData(waterVolume[PREVIOUS_MEASURES], waterVolume[ACTUAL_MEASURES]);

            lastTimeStamp += WAITTIME;
            sendResultsToDatabase("TankCalculations",resFuelLevel, resFuelVol, resFuelTemperature, resWaterLevel, resWaterVolume);


            /* for (int i = 0; i < results.Length; i++)
             {
                 Console.WriteLine("Calc: " + results[i]);
                 //Console.WriteLine("ACT: " + actualMeasures[i]);
             }
           

             for (int i=0;i<previousMeasures.Length;i++)
                 Console.WriteLine("OBLICZONE: " + previousMeasures[i]);*/
                
           
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
        public static void calculateDataWithCudafy(GThread thread, float[] prevMeasures, float[] actMeasures)
        {
           int tid = thread.blockIdx.x;
           if (tid < prevMeasures.Length)
            {
                if ((prevMeasures[tid] -= actMeasures[tid]) != 0)
                    prevMeasures[tid] /= WAITTIME;
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
