"""Ten NAB streams, identified by matching the fastness values recorded in the
submitted manuscript (Table 8) against every series in the NAB repository.
Each matched to 3 decimal places, so the identification is exact."""
import pandas as pd, numpy as np
R = "nab_repo/data/"
NAB = {
 "taxi":             R+"realKnownCause/nyc_taxi.csv",
 "traffic":          R+"realTraffic/speed_7578.csv",
 "cpu":              R+"realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
 "machine_temp":     R+"realKnownCause/machine_temperature_system_failure.csv",
 "grok_asg_anomaly": R+"realAWSCloudwatch/grok_asg_anomaly.csv",
 "rds_cpu_util":     R+"realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
 "iio_networkin":    R+"realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
 "twitter_aapl":     R+"realTweets/Twitter_volume_AAPL.csv",
 "exchange_cpc":     R+"realAdExchange/exchange-2_cpc_results.csv",
 "occupancy":        R+"realTraffic/occupancy_6005.csv",
}
def load(name, split0=0.50):
    v = pd.read_csv(NAB[name])["value"].to_numpy(float)
    A = v[:, None]                       # ONE continuous stream, N=1
    # tau is normalised for NAB; the unit is the TRAINING-span standard
    # deviation, so the tolerance never sees validation or test data.
    unit = float(A[:int(split0*len(A))].std())
    return A, unit
def fastness(A):
    v = A[:,0]; return float(np.std(np.diff(v))/np.std(v))
