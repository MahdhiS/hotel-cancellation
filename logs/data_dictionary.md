# Data Dictionary

IT3091 Machine Learning, Group 2026-DS-06K. Exported from the team sheet (same as Table 4 in the final report).

The **Known?** column asks: is this value known at the moment the booking is made? Our model must predict at booking time, but the dataset values were saved one day before arrival (Antonio et al., 2019), so any value that can change after booking may contain later information.

**Use:** Allowed, Banned (leakage), Excluded (fairness), Target or ID.

| Column | Meaning | Type | Example values | Known? | Use | Reason / doubts |
|---|---|---|---|---|---|---|
| booking_id | Row number from the raw file, used as the booking ID | ID | 0, 1, 2 | - | ID | The data has no ID of its own. Kept in every saved file; never a model input (D-14). |
| is_canceled | Booking cancelled (1) or not (0). No-shows are also 1 | Binary | 0, 1 | No | Target | This is what we predict. |
| hotel | City Hotel or Resort Hotel | Category | City Hotel, Resort Hotel | Yes | Allowed | Chosen when booking. |
| lead_time | Days between booking date and arrival date | Number | 0, 34, 342 | Yes | Allowed | Calculated when the booking is made. |
| arrival_date_year | Year of arrival | Category | 2015, 2016, 2017 | Yes | Allowed | The arrival date is chosen when booking. |
| arrival_date_month | Month of arrival | Category | January, February | Yes | Allowed | As above. |
| arrival_date_week_number | Week number of arrival | Category | 27, 28, 29 | Yes | Allowed | As above. |
| arrival_date_day_of_month | Day of the month of arrival | Category | 1, 2, 3 | Yes | Allowed | As above. |
| stays_in_weekend_nights | Weekend nights (Sat or Sun) booked | Integer | 0, 1, 2 | Yes | Allowed | Part of the booked stay. |
| stays_in_week_nights | Week nights (Mon to Fri) booked | Integer | 0, 2, 5 | Yes | Allowed | Part of the booked stay. |
| adults | Number of adults | Integer | 1, 2, 3 | Yes | Allowed | Entered when booking. |
| children | Number of children | Number | 0, 1, 2 | Yes | Allowed | Stored as a decimal because 4 values were missing in the raw data. |
| babies | Number of babies | Integer | 0, 1 | Yes | Allowed | Entered when booking. |
| meal | Meal package: BB = Bed & Breakfast, HB = Half Board, FB = Full Board, SC / Undefined = no meal package | Category | BB, HB, SC | Yes | Allowed | Chosen when booking. |
| market_segment | How the booking was made. TA = Travel Agent, TO = Tour Operator | Category | Online TA, Direct, Groups | Yes | Allowed | Known when the booking is made. |
| distribution_channel | Channel the booking came through | Category | TA/TO, Direct, Corporate | Yes | Allowed | Known when the booking is made. |
| is_repeated_guest | 1 = the guest has stayed before, 0 = not | Binary | 0, 1 | Yes | Allowed | History before this booking. |
| previous_cancellations | Earlier bookings by this guest that were cancelled | Integer | 0, 1, 2 | Yes | Allowed | History before this booking. |
| previous_bookings_not_canceled | Earlier bookings by this guest that were not cancelled | Integer | 0, 1, 2 | Yes | Allowed | History before this booking. |
| reserved_room_type | Room type booked (code) | Category | A, D, E | Yes | Allowed | Chosen when booking. |
| deposit_type | Deposit given: No Deposit; Non Refund (paid, no refund); Refundable | Category | No Deposit, Non Refund, Refundable | Yes | Allowed | Agreed when booking. Our deposit policy changes this column (see Section 3.5). |
| agent | ID of the travel agency (0 = no agent) | Category | 0, 9, 240 | Yes | Allowed | An ID code, so treated as a category, not a number. |
| company | ID of the company that made or pays for the booking (0 = no company) | Category | 0, 40, 223 | Yes | Allowed | An ID code. 94.1% of bookings are 0. |
| adr | Average daily rate: average price per night | Number | 0, 98.4, 134 | Yes | Allowed | The price is agreed when booking. Doubt: the value is saved one day before arrival, so later price changes may be included. |
| reservation_status | Final status of the booking | Category | Check-Out, Canceled, No-Show | No | Banned | This is the outcome itself (D-7). |
| reservation_status_date | Date the final status was set | Date | 2015-07-01 | No | Banned | Set after the outcome is known (D-7). |
| assigned_room_type | Room type the hotel actually gave | Category | A, D, E | No | Banned | Set near check-in and changes with upgrades (D-15). |
| booking_changes | Number of changes made to the booking | Integer | 0, 1, 2 | No | Banned | Counts changes made after booking, saved the day before arrival (D-15). |
| days_in_waiting_list | Days the booking waited before the hotel confirmed it | Integer | 0, 50, 150 | No | Banned | The final value is only known after the booking request (D-15). |
| customer_type | Transient, Transient-Party, Contract or Group | Category | Transient, Contract | No | Banned | Doubt: can change after booking, for example when the booking is linked to a group; saved the day before arrival (D-15). |
| required_car_parking_spaces | Parking spaces asked for | Integer | 0, 1 | No | Banned | Guests can add parking after booking (D-5). |
| total_of_special_requests | Special requests, e.g. twin bed, high floor | Integer | 0, 1, 2 | No | Banned | Guests can add requests after booking (D-5). |
| country | Guest's country (ISO code) | Category | PRT, GBR, FRA | Unsure | Excluded | Not used, so no deposit rule judges a guest by nationality; kept only for the fairness comparison (D-16). Doubt: may be corrected at check-in. |

Of the 32 original columns: 1 target, 22 allowed, 8 banned for leakage and 1 excluded for fairness. `booking_id` was added from the row number of the raw file.
