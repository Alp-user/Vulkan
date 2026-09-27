struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

struct VSInput {
    float3 pos;
    float3 color;
    float2 in_uv;
};

struct VSOutput {
    float3 color;
    float4 sv_position : SV_Position;
    float2 out_uv;
};

ConstantBuffer<UniformBuffer> ubo;
Sampler2D texture;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.sv_position = mul(ubo.projection, mul(ubo.view, mul(ubo.model, float4(input.pos, 1.0))));
    output.out_uv = input.in_uv;
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput in_vertex) : SV_Target {
    return texture.Sample(in_vertex.out_uv * 2.0f);
}
