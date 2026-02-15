/* Auto-generated — do not edit.
 * Source: NX-AI/xlstm reference (vanilla backend)
 * Regenerate: make reference
 */

#ifndef REFERENCE_DATA_H_
#define REFERENCE_DATA_H_

// ========================================================================
// sLSTM reference data
// ========================================================================

// Test 1: Single timestep, zero initial state
// B=1, T=1, I=2, H=2
const float kTest1_W[] = {0.96345764f, 0.74364203f, 0.45035860f, -1.05276048f, 0.33920923f, -0.61727244f, -0.02153374f, -0.80233347f, -0.37606764f, 0.82436150f, -0.19623932f, -0.70180357f, -0.36394066f, -0.27971509f, -0.38441944f, 0.38122270f};
const float kTest1_R[] = {0.82115847f, -0.07979874f, -0.24869877f, 0.21979463f, -0.37906557f, 0.53915882f, 0.40040028f, 0.84031028f, 0.63956219f, 0.64821142f, 0.30523324f, 0.66736889f, -0.11581216f, 0.02087975f, -0.12578765f, 0.42992926f};
const float kTest1_b[] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};
const float kTest1_input[] = {1.00000000f, 0.50000000f};
const float kTest1_expected_y[] = {0.01359604f, -0.22511525f};
const float kTest1_expected_c[] = {0.03609742f, -0.49837443f};
const float kTest1_expected_n[] = {1.00000000f, 1.00000000f};
const float kTest1_expected_m[] = {1.33527863f, -0.07602164f};

// Test 2: 3 timesteps, state propagation (B=1, T=3, I=2, H=2)
const float kTest2_input[] = {1.00000000f, 0.50000000f, 0.30000001f, -0.20000000f, -0.50000000f, 1.00000000f};
const float kTest2_expected_y[] = {0.16812575f, -0.20084262f};
const float kTest2_expected_c[] = {0.63085783f, -0.64888012f};
const float kTest2_expected_n[] = {1.78957176f, 2.04787779f};
const float kTest2_expected_m[] = {0.21861291f, -0.93307185f};
const float kTest2_expected_output[] = {0.01359604f, -0.22511525f, -0.05999865f, -0.07478932f, 0.16812575f, -0.20084262f};

// Test 3: Large inputs, overflow prevention
// B=1, T=1, I=2, H=2
// i_raw = 100 — would overflow without m-stabilizer
const float kTest3_W[] = {5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f};
const float kTest3_R[] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};
const float kTest3_b[] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};
const float kTest3_input[] = {10.00000000f, 10.00000000f};
const float kTest3_expected_y[] = {0.99995458f, 0.99995458f};
const float kTest3_expected_c[] = {1.00000000f, 1.00000000f};
const float kTest3_expected_n[] = {1.00000000f, 1.00000000f};
const float kTest3_expected_m[] = {100.00000000f, 100.00000000f};

// ========================================================================
// mLSTM reference data
// ========================================================================

// mLSTM Test 1: Single timestep, zero initial state
// B=1, T=1, I=3, H=2
const float kMTest1_W[] = {0.16868509f, -0.08888861f, -0.15176380f, -0.29400593f, 0.17430259f, 0.33017048f, -0.10981881f, -0.18958491f, 0.38355353f, -0.59625101f, 0.34917596f, -0.70486146f, 0.08968981f, 0.94757402f, 0.68445408f, -0.80163509f, -0.66247869f, 0.08921422f, -1.06687653f, 0.52617890f, -0.19424154f, -0.46717295f, 0.91594690f, -0.16891940f, 0.44027215f, 0.77707928f, 0.31330803f, -0.08774360f, 0.04914189f, -0.04675372f};
const float kMTest1_b[] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};
const float kMTest1_input[] = {1.00000000f, 0.50000000f, -0.30000001f};
const float kMTest1_expected_y[] = {0.00081466f, -0.00190351f};
const float kMTest1_expected_C[] = {-0.07526524f, 0.24370453f, -0.04949084f, 0.16024849f};
const float kMTest1_expected_n[] = {-0.21015556f, -0.13818829f};
const float kMTest1_expected_m[] = {-0.67262405f};

// mLSTM Test 2: 3 timesteps, state propagation (B=1, T=3, I=3, H=2)
const float kMTest2_input[] = {1.00000000f, 0.50000000f, -0.30000001f, 0.30000001f, -0.20000000f, 0.80000001f, -0.50000000f, 1.00000000f, 0.10000000f};
const float kMTest2_expected_y[] = {0.25203866f, -0.05591988f};
const float kMTest2_expected_C[] = {-0.05742935f, 0.02888295f, 0.36041993f, -0.09147088f};
const float kMTest2_expected_n[] = {-0.04624049f, 0.31464157f};
const float kMTest2_expected_m[] = {1.04019296f};
const float kMTest2_expected_output[] = {0.00081466f, -0.00190351f, -0.01144831f, 0.00183737f, 0.25203866f, -0.05591988f};

// mLSTM Test 3: Large values, overflow prevention
// B=1, T=1, I=3, H=2
// i_raw = 150 — would overflow without m-stabilizer
const float kMTest3_W[] = {0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 5.00000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f, 0.50000000f};
const float kMTest3_b[] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};
const float kMTest3_input[] = {10.00000000f, 10.00000000f, 10.00000000f};
const float kMTest3_expected_y[] = {14.99999332f, 14.99999332f};
const float kMTest3_expected_C[] = {159.09902954f, 159.09902954f, 159.09902954f, 159.09902954f};
const float kMTest3_expected_n[] = {10.60660172f, 10.60660172f};
const float kMTest3_expected_m[] = {150.00000000f};

#endif /* REFERENCE_DATA_H_ */
