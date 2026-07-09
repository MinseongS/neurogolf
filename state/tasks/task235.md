
## ADOPTED 20260709T041314Z
- cost: 170 -> 129 (points 20.1402)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task235.onnx
- note: min-merge from nets

## RULE (recovered 2026-07-09, public autopsy)
- 입력 4×14 = 회색(5) 4×4 패널 3개(검은 열 구분). 각 패널에 2×2 검은 구멍(5개 위치 클래스).
- 출력 3×3: row i = 패널 i의 구멍 클래스를 색으로 인코딩(2,3,4,8 관찰).
- 채택 넷 메커니즘: negative-pad strided Conv sampler-decoder(strides=[30,5], pads=[-1,0,0,-14],
  커널 = 답 색과 같은 linear code — fitted 상수, 복사 금지) + BitwiseXor(code, channel_ids) →
  grouped ConvInteger(w=-1, zero_point=1) free-output renderer. insights.yaml:
  negpad_strided_conv_sampler_decoder, xor_channel_id_convinteger_renderer.
