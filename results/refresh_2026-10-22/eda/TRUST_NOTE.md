# Why `tank_trust.json` here is the published study's

The trust tier is a mass-balance test: each hour, the change in tank level must equal inflow
minus outflow. Waltr has since revised how it derives flows. In the re-downloaded data, many hours
report zero inflow and zero outflow while the level moved (TECH_PARK: 67 % of hours, median
level change 0.13 KL), so the identity fails by construction rather than because a sensor
degraded. Recomputed on the revised data, the test marks 18 of 24 tanks degraded
(`tank_trust_revised_flows.json`). That measures Waltr's processing, not sensor health.

`tank_trust.json` is therefore the tier assigned on the original, self-consistent data
(`eda/tank_trust.json`): same physical sensors, assessed when the identity still held.
