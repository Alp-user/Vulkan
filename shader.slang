struct VSInput
{
  float2 pos;
  float3 color;
};

struct VSOutput
{
  float3 color;
  float4 sv_position : SV_Position;
};

[shader("vertex")]
VSOutput vertMain(VSInput input){
  VSOutput output;
  output.sv_position = float4(input.pos, 0.0, 1.0);
  output.color = input.color;
  return output;
}

[shader("fragment")]
float4 fragMain(VSOutput in_vertex) : SV_Target {
  return float4(in_vertex.color, 1.0);
}
