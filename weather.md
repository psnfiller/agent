# Goal

Show Met Éireann weather warnings for Wicklow and Dublin on the wall dashboard (/dashboard-wall/stuff), and also show the full Met Éireann forecast similar to the big weather panel.

This gives you two clear warnings panels (Wicklow + Dublin) and a full forecast card.

Approach
- Use Home Assistant’s Meteoalarm integration to surface Met Éireann warnings as binary_sensors (one per county).
- Use the HACS “meteoalarm-card” to render those warnings clearly.
- Use the core Met Éireann integration for the forecast with the stock weather-forecast card.

Steps
1) Add Meteoalarm (warnings)
   - HA UI: Settings → Devices & Services → + Add Integration → Meteoalarm
   - Create two entries: one for County Wicklow, one for County Dublin.
   - You should get entities similar to:
     - binary_sensor.meteoalarm_wicklow
     - binary_sensor.meteoalarm_dublin

2) Add Met Éireann weather (forecast)
   - Settings → Devices & Services → + Add Integration → Met Éireann (Ireland)
   - Note the weather entity created, e.g. weather.met_eireann_home (your ID may differ).

3) Install the warnings card (nice UI)
   - HACS → Frontend → Explore & Download → search “meteoalarm-card” → Install
   - Restart if prompted.

4) Add cards to the wall dashboard Stuff view
   - Open /dashboard-wall, edit the view whose path is stuff (or create one with that path).
   - Add a Manual card and paste the YAML below (adjust entity IDs to match your setup):

   # Warnings for Wicklow + Dublin
   type: horizontal-stack
   cards:
     - type: custom:meteoalarm-card
       name: Wicklow Warnings
       entity: binary_sensor.meteoalarm_wicklow
       show_headline: true
       description: true
       hide_when_no_warning: true
     - type: custom:meteoalarm-card
       name: Dublin Warnings
       entity: binary_sensor.meteoalarm_dublin
       show_headline: true
       description: true
       hide_when_no_warning: true

   # Full Met Éireann forecast (standard card)
   - type: weather-forecast
     name: Met Éireann Forecast
     entity: weather.met_eireann_home   # change to your actual entity id

Optional: Direct Met Éireann warnings via HACS
- If you’d rather pull warnings from a dedicated HACS integration:
  1) HACS → Integrations → Explore & Download → “Met Éireann Weather Warnings” → Install
  2) Add two instances via Settings → Devices & Services (Wicklow, Dublin)
  3) Point the meteoalarm-card at the new binary_sensor IDs it creates.

Notes
- Entity IDs may differ; confirm under Settings → Devices & Services → Entities.
- The meteoalarm-card gives the best UX (levels, colors, icons, headlines). If you don’t install it, a simple Markdown card can list warning attributes, but it’s less readable.
- For the forecast, the stock weather-forecast card is simple and tablet-friendly; you can also add the “Weather Forecast” custom card if you prefer alternative layouts.
