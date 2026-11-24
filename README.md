# AACE-CM experiment code

This repository accompanies the AACE-CM experiments on physics-enhanced predictive maintenance. The experiments start with operating-condition identification and fault augmentation, then evaluate equipment health assessment, degradation forecasting, and fault identification. The final experiments compare simulated maintenance policies and examine how expert preferences and equipment evidence support the review of maintenance recommendations.

## Experiment modules

| Experiment | Implementation and evaluation | Code |
| --- | --- | --- |
| Operating-condition identification | Compares rule-based, learned, and hybrid configurations against reviewed operating-condition labels. | [01_operating_conditions](aace_cm/01_operating_conditions/) |
| Physics-based fault augmentation | Constructs mechanism-based fault scenarios on measured production backgrounds. The augmentation models changes in vibration and temperature. | [02_fault_augmentation](aace_cm/02_fault_augmentation/) |
| Dual-space anomaly detection | Learns a normal-operation reference in latent and residual spaces, compares detector configurations, and evaluates condition-calibrated alarms on held-out cases. | [03_anomaly_detection](aace_cm/03_anomaly_detection/) |
| Health-score forecasting | Compares forecasts across lookback lengths and prediction horizons. A backbone ablation examines the contribution of individual components. | [04_health_forecasting](aace_cm/04_health_forecasting/) |
| Fault identification | Uses time-frequency residual features to discover fault groups before assigning engineering names. A separate evaluation withholds fault mechanisms to assess recognition of unfamiliar patterns. | [05_fault_identification](aace_cm/05_fault_identification/) |
| Maintenance-policy comparison | Compares rule-based and reinforcement learning policies in simulation using reward, missed interventions, and intervention frequency. | [06_maintenance_policies](aace_cm/06_maintenance_policies/) |
| Agent-in-the-loop evaluation | Configures bounded expert preferences, compares alternative actions through counterfactual replay, and supports evidence-based review of recorded QR-DQN recommendations. | [07_agent_in_the_loop](aace_cm/07_agent_in_the_loop/) |
