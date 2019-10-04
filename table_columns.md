### Column Table
| Original Column Name | New Column Name  | Data Type |
|----------------|-----------|-------------|
| Date of Interview | date | discrete (mm/dd/yyyy) |
| Client name | client | categorical (16 categories) |
| Industry | industry | categorical (8 categories) |
| Location | location | categorical (12 categories) |
| Position to be closed | position | categorical (8 categories) |
| Nature of Skillset | skillset | categorical (93 categories) |
| Interview Type | interview_type | categorical (7 categories) |
| Name(Cand ID) | (dropped) | discrete (string) |
| Gender | gender | categorical (3) | 1 NaN value | 
| Candidate Current Location | cand_cur_loc | categorical (12 categories) |
| Candidate Job Location | cand_job_loc | categorical (8 categories) | 
| Interview Venue | interview_loc | categorical (8 locations) |
| Candidate Native location | (dropped) categorical (47 categories) |
| Marital status | mar_status | categorical (2 categories) |
| "Have you obtained the necessary permission to start at the required time?" | start_perm | categorical (9 categories) | 
| Hope there will be no unscheduled meetings | unsch_mtgs | categorical (9 categories) |
| "Can I call you three hours before the interview and follow up on your attendance for the interview?" | precall | categorical (7 categories) | 
| Can I have an alternative number/ desk number? I assure you that I will not trouble you too much. | alt_num categorical (8 categories) |
| "Have you taken a printout of your updated resume. Have you read the JD and understood the same?" | res_jd | categorical (10 categories) |  
| "Are you clear with the venue details and the landmark?" | venue_clear |  categorical (9 categories) |
| "Has the call letter been shared?" | letter_shared | categorical (14 categories) | 
| Expected Attendance | exp_attend | categorical (2 categories) |
| Observed Attendance | obs_attend | categorical (2 categories) |