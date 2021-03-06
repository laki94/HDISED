﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Globalization;
using System.Threading;

namespace HDISED_GPU
{
    public class SQL
    {
        public const int WAITTIME = 10;

        public const string DATADB = "Data";
        public const string RESULTDB = "Calculations";
        public const string ARCHIVESDB = "Archives";
        public const string SERVER = "localhost";
        public const string USER = "sa";
        public const string PASSWORD = "root";

        public static List<int> tanks;
        public static List<int> nozzles;
        public static int lastTimeStamp = 100;
        public static SqlConnection archivesConnection;
        public static SqlConnection dataConnection;
        public static SqlConnection resultsConnection;

        public static CudafyModule km;
        public static GPGPU gpu;

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
            float[] result = new float[tanks.Count];
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
            string[] result = new string[tanks.Count];

            for (int id = 1; id <= tanks.Count; id++)
            {
                tmpQuery.CommandText += " and TankId = " + id;
                if (reader.Read())
                    result[id - 1] = reader.GetValue(0).ToString();
                else
                {
                    reader.Close();
                    return null;
                }
                tmpQuery.CommandText = orgCommText;
            }
            reader.Close();
            return result;
        }

        public static string[] executeForEveryNozzle(SqlCommand _query)
        {
            var tmpQuery = _query;
            string orgCommText = _query.CommandText;
            var reader = tmpQuery.ExecuteReader();
            string[] result = new string[nozzles.Count];

            for (int id = 1; id <= nozzles.Count; id++)
            {
                tmpQuery.CommandText += " and NozzleId = " + id;
                if (reader.Read())
                    result[id - 1] = reader.GetValue(0).ToString();
                else
                {
                    reader.Close();
                    return null;
                }
                tmpQuery.CommandText = orgCommText;
            }
            reader.Close();
            return result;
        }

        private static List<int> getItems(SqlConnection connection, string table, string parameter)
        {
            SqlCommand query = new SqlCommand("Select " + parameter + " from " + table + " group by " + parameter + " order by " + parameter);
            query.Connection = connection;
            List<int> tmpList = new List<int>();

            Object tmp = new Object();
            var reader = query.ExecuteReader();

            for (int i = 0; reader.Read(); i++)
                if (!reader.IsDBNull(0))
                    tmpList.Add(reader.GetInt32(0));
            
            reader.Close();
            return tmpList;
        }

        private static DateTime getDate(SqlConnection connection, string table, int id)
        {
            SqlCommand query = new SqlCommand("Select date from " + table + " where TimeId = " + id);
            query.Connection = connection;
            var reader = query.ExecuteReader();
            reader.Read();
            var result = reader.GetDateTime(0);
            reader.Close();
           
            return result;
        }

        private static int getLastDate(SqlConnection connection, string table)
        {
            SqlCommand query = new SqlCommand("Select top (1) TimeId from " + table + " order by TimeId desc");
            query.Connection = connection;
            var reader = query.ExecuteReader();
            reader.Read();
            var result = reader.GetInt32(0);
            reader.Close();

            return result;
        }

        private static int[][] getPrevAndActValuesInt(SqlConnection dataConnection, string table, string parameter, int length, bool isNozzleIdNecessary = false)
        {
            int[][] result = new int[2][];
            string[] tmpPrev;
            string[] tmpAct;
            if (isNozzleIdNecessary)
            {
                tmpPrev = executeForEveryNozzle(getParameterCommand(table, parameter, lastTimeStamp));
                tmpAct = executeForEveryNozzle(getParameterCommand(table, parameter, lastTimeStamp + WAITTIME));
            }
            else
            {
                tmpPrev = executeForEveryTank(getParameterCommand(table, parameter, lastTimeStamp));
                tmpAct = executeForEveryTank(getParameterCommand(table, parameter, lastTimeStamp + WAITTIME));
            }

            if ((tmpPrev != null) && (tmpAct != null))
            {
                int[] previousMeasures = new int[length];
                int[] actualMeasures = new int[length];

                for (int i = 0; i < previousMeasures.Length; i++)
                {
                    previousMeasures[i] = Int32.Parse(tmpPrev[i]) >= 0 ? Int32.Parse(tmpPrev[i]) : 0;
                    actualMeasures[i] = Int32.Parse(tmpAct[i]) > 0 ? Int32.Parse(tmpAct[i]) : 0;
                }

                result[0] = previousMeasures;
                result[1] = actualMeasures;
                return result;
            }
            else
                return null;
        }

        private static float[][] getPrevAndActValuesFloat(SqlConnection dataConnection,string table, string parameter, int length, bool isNozzleIdNecessary = false)
        {
            float[][] result = new float[2][];
            string[] tmpPrev;
            string[] tmpAct;
            if (isNozzleIdNecessary)
            {
                tmpPrev = executeForEveryNozzle(getParameterCommand(table, parameter, lastTimeStamp));
                tmpAct = executeForEveryNozzle(getParameterCommand(table, parameter, lastTimeStamp + WAITTIME));
            }
            else
            {
                tmpPrev = executeForEveryTank(getParameterCommand(table, parameter, lastTimeStamp));
                tmpAct = executeForEveryTank(getParameterCommand(table, parameter, lastTimeStamp + WAITTIME));
            }

            if ((tmpPrev != null) && (tmpAct != null))
            {
                float[] previousMeasures = new float[length];
                float[] actualMeasures = new float[length];

                for (int i = 0; i < previousMeasures.Length; i++)
                {
                    previousMeasures[i] = float.Parse(tmpPrev[i]) >= 0 ? float.Parse(tmpPrev[i]) : 0;
                    actualMeasures[i] = float.Parse(tmpAct[i]) >= 0 ? float.Parse(tmpAct[i]) : 0;
                }
                result[0] = previousMeasures;
                result[1] = actualMeasures;
                return result;
            }
            else
                return null;
        }



        public static void prepareGPU()
        {
            km = CudafyTranslator.Cudafy(); 
            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);
        }

        public static float[] prepareAndCalculateFloatData(float[] prevMeasures, float[] actMeasures)
        {
            if ((prevMeasures != null) && (actMeasures != null))
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
            else
                return null;
        }

        public static int[] prepareAndCalculateIntData(int[] prevMeasures, int[] actMeasures)
        {
            if ((prevMeasures != null) && (actMeasures != null))
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
            else
                return null;
        }

        public static void sendTankMeasuresResultsToDatabase(string table, int[] fuelLevel = null, float[] fuelVolume = null, int[] fuelTemperature = null, int[] waterLevel = null, float[] waterVolume = null)
        {
            var tmpQuery = "INSERT INTO " + table + " VALUES (";
            var calcConnection = connectToServer(SERVER, RESULTDB, USER, PASSWORD);
            SqlCommand insertData = new SqlCommand(tmpQuery);
            NumberFormatInfo nfi = new NumberFormatInfo();
            nfi.NumberDecimalSeparator = ".";

            for (int id = 0; id < tanks.Count; id++) 
            {
                insertData.CommandText += "'" + getDate(dataConnection, "Time", lastTimeStamp) + "'" + ", " + "'" + getDate(dataConnection, "Time", (lastTimeStamp + WAITTIME)) + "'" + ", " + tanks[id];
                if (fuelLevel != null)
                    insertData.CommandText += ", " + fuelLevel[id]; 
                else
                    insertData.CommandText += ", " + 0; 

                if (fuelVolume != null)
                    insertData.CommandText += ", " + fuelVolume[id].ToString(nfi);
                else
                    insertData.CommandText += ", " + 0; 

                if (fuelTemperature != null)
                    insertData.CommandText += ", " + fuelTemperature[id];
                else
                    insertData.CommandText += ", " + 0; 

                if (waterLevel != null)
                    insertData.CommandText += ", " + waterLevel[id];
                else
                    insertData.CommandText += ", " + 0; 

                if (waterVolume != null)
                    insertData.CommandText += ", " + waterVolume[id].ToString(nfi);
                else
                    insertData.CommandText += ", " + 0; 

                insertData.CommandText += ")";
                insertData.Connection = calcConnection;
                insertData.ExecuteNonQuery();
                insertData.CommandText = tmpQuery;   
            }
            calcConnection.Close();
        }

        public static void sendNozzleMeasuresResultsToDatabase(string table, float[] totalCounter = null)
        {
            var tmpQuery = "INSERT INTO " + table + " VALUES (";
            var calcConnection = connectToServer(SERVER, RESULTDB, USER, PASSWORD);
            SqlCommand insertData = new SqlCommand(tmpQuery);
            NumberFormatInfo nfi = new NumberFormatInfo();
            nfi.NumberDecimalSeparator = ".";

            for (int id = 0; id < nozzles.Count; id++)
            {
                insertData.CommandText += "'" + getDate(dataConnection, "Time", lastTimeStamp) + "'" + ", " + "'" + getDate(dataConnection, "Time", (lastTimeStamp + WAITTIME)) + "'" + ", " + nozzles[id];

                if (totalCounter != null)
                    insertData.CommandText += ", " + totalCounter[id].ToString(nfi);
                else
                    insertData.CommandText += ", " + 0;

                insertData.CommandText += ")";
                insertData.Connection = calcConnection;
                insertData.ExecuteNonQuery();
                insertData.CommandText = tmpQuery;
            }
            calcConnection.Close();
        }

        public static void Execute(int waitTime = WAITTIME)
        {
            const int PREVIOUS_MEASURES = 0;
            const int ACTUAL_MEASURES = 1;
            dataConnection = connectToServer(SERVER, DATADB, USER, PASSWORD);

            int lastDBUpdate = getLastDate(dataConnection, "Time");

            prepareGPU();

            while ((lastTimeStamp + waitTime) < lastDBUpdate)
            {
                tanks = getItems(dataConnection, "TankMeasures", "TankId");

                float[][] fuelVolumeMeasures = getPrevAndActValuesFloat(dataConnection, "TankMeasures", "FuelVolume", tanks.Count);
                float[] resFuelVol = prepareAndCalculateFloatData(fuelVolumeMeasures[PREVIOUS_MEASURES], fuelVolumeMeasures[ACTUAL_MEASURES]);

                int[][] fuelLevelMeasures = getPrevAndActValuesInt(dataConnection, "TankMeasures", "FuelLevel", tanks.Count);
                int[] resFuelLevel = prepareAndCalculateIntData(fuelLevelMeasures[PREVIOUS_MEASURES], fuelLevelMeasures[ACTUAL_MEASURES]);

                int[][] fuelTemperature = getPrevAndActValuesInt(dataConnection, "TankMeasures", "FuelTemperature", tanks.Count);
                int[] resFuelTemperature = prepareAndCalculateIntData(fuelTemperature[PREVIOUS_MEASURES], fuelTemperature[ACTUAL_MEASURES]);

                int[][] waterLevel = getPrevAndActValuesInt(dataConnection, "TankMeasures", "WaterLevel", tanks.Count);
                int[] resWaterLevel = prepareAndCalculateIntData(waterLevel[PREVIOUS_MEASURES], waterLevel[ACTUAL_MEASURES]);

                float[][] waterVolume = getPrevAndActValuesFloat(dataConnection, "TankMeasures", "WaterVolume", tanks.Count);
                float[] resWaterVolume = prepareAndCalculateFloatData(waterVolume[PREVIOUS_MEASURES], waterVolume[ACTUAL_MEASURES]);

                nozzles = getItems(dataConnection, "NozzleMeasures", "NozzleId");

                float[][] totalCounter = getPrevAndActValuesFloat(dataConnection, "NozzleMeasures", "TotalCounter", nozzles.Count, true);
                float[] resTotalCounter = prepareAndCalculateFloatData(totalCounter[PREVIOUS_MEASURES], totalCounter[ACTUAL_MEASURES]);

                sendTankMeasuresResultsToDatabase("TankCalculations", resFuelLevel, resFuelVol, resFuelTemperature, resWaterLevel, resWaterVolume);
                sendNozzleMeasuresResultsToDatabase("NozzleCalculations", resTotalCounter);

                lastTimeStamp += waitTime;
                tanks.Clear();
                nozzles.Clear();

                Console.WriteLine(lastTimeStamp + " - " + lastDBUpdate);
            }
            Console.WriteLine("Nie mozna agregowac dalej, brak nowych danych dla podanego przedzialu czasowego " + lastTimeStamp + " / " + lastDBUpdate);
            if (archivesConnection != null) 
                archivesConnection.Close();

            if (dataConnection != null)
                dataConnection.Close();

            if (resultsConnection != null)
            resultsConnection.Close();

            Console.WriteLine("Czekanie przez " + waitTime * 60000 + "s ...");
            Thread.Sleep(waitTime * 60000);
            Execute(waitTime);
        }

        [Cudafy]
        public static void calculateDataWithCudafy(GThread thread, float[] prevMeasures, float[] actMeasures)
        {
            int tid = thread.blockIdx.x;
            if (tid < prevMeasures.Length)
            {
                prevMeasures[tid] -= actMeasures[tid];
                if (prevMeasures[tid] < 0) 
                    prevMeasures[tid] = 0;
                if (prevMeasures[tid] != 0)
                    prevMeasures[tid] /= WAITTIME;
            }
        }    
    }
}
