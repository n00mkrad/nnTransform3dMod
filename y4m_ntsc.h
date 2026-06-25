#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

struct LdJsonFieldMeta {
    int audioSamples = 0;
    int fieldPhaseID = 0;
    bool isFirstField = false;
    int medianBurstIRE = 20;
    bool pad = false;
    int seqNo = 0;
    int syncConf = 100;
};

struct LdJsonVideoParameters {
    int activeVideoStart = 0;
    int activeVideoEnd = 0;
    int black16bIre = 0;
    int white16bIre = 0;
    int colourBurstStart = 0;
    int colourBurstEnd = 0;
    int fieldWidth = 0;
    int fieldHeight = 0;
    std::string system;
    double sampleRate = 0.0;
    double chromaGain = -1.0;
    bool isWidescreen = false;
};

struct LdJsonMetadata {
    LdJsonVideoParameters videoParameters;
    std::vector<LdJsonFieldMeta> fields;
};

struct Y4mNtscConfig {
    bool fullFrame = false;
    bool forceLimited = false;
    bool generatedFieldMetadata = false;
    bool levelsOverride = false;
    int black16bIreOverride = 0;
    int white16bIreOverride = 0;
    int activeVideoStartOverride = -1;
    int activeVideoEndOverride = -1;
    int firstLine = 40;
    int lastLine = 525;
    std::size_t frameIndexOffset = 0;
    double chromaGain = 1.0;
};

bool loadLdJsonMetadata(const std::string& path, LdJsonMetadata& metadata, std::string& error, bool ignoreFields = false);
void initializeDefaultNtscMetadata(LdJsonMetadata& metadata);
bool appendGeneratedFieldMetadata(LdJsonMetadata& metadata, std::size_t fieldCount, std::string& error);

class Y4mNtscWriter {
public:
    Y4mNtscWriter(const LdJsonMetadata& metadata, const Y4mNtscConfig& config, std::ostream& output);
    void writeFrame(const std::vector<uint16_t>& lumaPlane, const std::vector<uint16_t>& chromaPlane);

    int getOutputWidth() const { return outputWidth; }
    int getOutputHeight() const { return outputHeight; }

private:
    static bool getLinePhase(int lineNumber, int firstFieldPhaseID, int secondFieldPhaseID);
    static std::size_t inferCyclePeriod(const std::vector<int>& phases);
    static std::string normalizeSystem(const std::string& system);

    uint16_t mapYToLimited(double y) const;
    uint16_t mapUToLimited(double u) const;
    uint16_t mapVToLimited(double v) const;
    double convolve5Tap(const std::vector<double>& data, int index) const;

    const LdJsonMetadata* metadata = nullptr;
    Y4mNtscConfig config;
    std::ostream* output = nullptr;
    std::vector<int> phaseCycle;
    std::size_t frameIndex = 0;

    int frameWidth = 0;
    int frameHeight = 0;
    int xStart = 0;
    int xEnd = 0;
    int yStart = 0;
    int yEnd = 0;
    int outputWidth = 0;
    int outputHeight = 0;

    double yOffset = 0.0;
    double yRange = 0.0;
    double yScale = 0.0;
    double cbScale = 0.0;
    double crScale = 0.0;
    double rotateSin = 0.0;
    double rotateCos = 0.0;
};
