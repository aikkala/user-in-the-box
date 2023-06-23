using UnityEngine;

namespace UserInTheBox
{
    public static class UitBUtils
    {

        public static string GetKeywordArgument(string argName)
        {
            // Get argName from command line arguments.
            // Throws an ArgumentException if argName is not found from command line arguments.
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == ("-" + argName) && (args.Length > i + 1))
                {
                    return args[i + 1];
                }
            }

            throw new System.ArgumentException("Could not find " + argName + " from command line arguments");
        }

        public static string GetOptionalKeywordArgument(string argName, string defaultValue)
        {
            if (GetOptionalArgument(argName))
            {
                return GetKeywordArgument(argName);
            }
            else
            {
                return defaultValue;
            }
        }

        public static bool GetOptionalArgument(string argName)
        {
            // Get argName from command line arguments.
            // Returns false if argName is not found.
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == ("-" + argName))
                {
                    return true;
                }
            }

            return false;
        }

        public static string GetStateString(SimulatedUser simulatedUser)
        {
            return TransformToString(simulatedUser.leftHandController.transform) + ", " +
                   TransformToString(simulatedUser.rightHandController.transform) + ", " +
                   TransformToString(simulatedUser.mainCamera.transform);

        }

        public static string GetStateHeader()
        {
            return "timestamp, " +
                   "left_pos_x, left_pos_y, left_pos_z, left_quat_x, left_quat_y, left_quat_z, left_quat_w, " +
                   "right_pos_x, right_pos_y, right_pos_z, right_quat_x, right_quat_y, right_quat_z, right_quat_w, " +
                   "head_pos_x, head_pos_y, head_pos_z, head_quat_x, head_quat_y, head_quat_z, head_quat_w";
        }
        
        public static string TransformToString(Transform transform, string delimiter=", ")
        {
            return Vector3ToString(transform.position, delimiter) + delimiter + 
                   QuaternionToString(transform.rotation, delimiter);
        }

        public static string Vector3ToString(Vector3 vec, string delimiter=", ")
        {
            return vec.x + delimiter + vec.y + delimiter + vec.z;
        }

        public static string QuaternionToString(Quaternion quat, string delimiter)
        {
            return quat.x + delimiter + quat.y + delimiter + quat.z + delimiter + quat.w;
        }
    }
}