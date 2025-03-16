//My shader modified by Me for use with obs-shaderfilter month/year v.02

//Section to converting GLSL to HLSL - can delete if unneeded
#define vec2 float2
#define vec3 float3
#define vec4 float4
#define ivec2 int2
#define ivec3 int3
#define ivec4 int4
#define mat2 float2x2
#define mat3 float3x3
#define mat4 float4x4
#define fract frac
#define mix lerp
#define iTime float
#define iTime elapsed_time
#define iResolution float4(uv_size,uv_pixel_interval)

/*
** Shaders have these variables pre loaded by the plugin **
** this section can be deleted if unneeded **

struct VertData {
    float4 pos : POSITION;
    float2 uv  : TEXCOORD0;
};

uniform float4x4 ViewProj;
uniform texture2d image;

uniform float elapsed_time;
uniform float2 uv_offset;
uniform float2 uv_scale;
uniform float2 uv_pixel_interval;
uniform float2 uv_size;
uniform float rand_f;
uniform float rand_instance_f;
uniform float rand_activation_f;
uniform int loops;
uniform float local_time;
*/

uniform texture2d gain_map;
uniform float gain_map_scale <
    string uiname="Gain Map Scale";
    float maximum=4.0;
    float minimum=0.0;
    float step=0.01;
    string widget_type="slider";
> = 4.0;

float4 mainImage(VertData v_in) : TARGET
{
    // For now, do a simple pass-through of the main image
    float4 baseColor = image.Sample(textureSampler, v_in.uv);
    float4 gain = gain_map.Sample(textureSampler, v_in.uv);
    return baseColor * gain * gain_map_scale;
}

/*
** Shaders use the built in Draw technique **
** this section can be deleted if unneeded **

technique Draw
{
    pass
    {
        vertex_shader = mainTransform(v_in);
        pixel_shader  = mainImage(v_in);
    }
}
*/
