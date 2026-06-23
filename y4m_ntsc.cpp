#include "y4m_ntsc.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "external/nlohmann/json.hpp"

namespace {
using json = nlohmann::json;

static constexpr double Y_MIN = 1.0 * 256.0;
static constexpr double Y_LEGAL_MIN = 16.0 * 256.0;
static constexpr double Y_ZERO = 16.0 * 256.0;
static constexpr double Y_SCALE = 219.0 * 256.0;
static constexpr double Y_LEGAL_MAX = 235.0 * 256.0;
static constexpr double Y_MAX = 254.75 * 256.0;
static constexpr double C_MIN = 1.0 * 256.0;
static constexpr double C_LEGAL_MIN = 16.0 * 256.0;
static constexpr double C_ZERO = 128.0 * 256.0;
static constexpr double C_SCALE = 112.0 * 256.0;
static constexpr double C_LEGAL_MAX = 240.0 * 256.0;
static constexpr double C_MAX = 254.75 * 256.0;
static constexpr double ONE_MINUS_Kb = 1.0 - 0.114;
static constexpr double ONE_MINUS_Kr = 1.0 - 0.299;
static constexpr double kB = 0.49211104112248356308804691718185;
static constexpr double kR = 0.87728321993817866838972487283129;
static constexpr double ROTATE_DEGREES = 33.0;
static constexpr double PI_D = 3.14159265358979323846;
}

bool loadLdJsonMetadata(const std::string& path, LdJsonMetadata& metadata, std::string& error) {
    try {
        std::ifstream is(path, std::ios::binary);
        if (!is.is_open()) {
            error = "Unable to open metadata JSON file: " + path;
            return false;
        }
        json root;
        is >> root;
        if (!root.contains("videoParameters") || !root["videoParameters"].is_object()) {
            error = "Metadata JSON is missing required object: videoParameters";
            return false;
        }
        const json& vp = root["videoParameters"];
        auto getInt = [&](const char* key) -> int {
            if (!vp.contains(key) || !vp[key].is_number_integer()) throw std::runtime_error(std::string("Missing/invalid integer key: videoParameters.") + key);
            return vp[key].get<int>();
        };
        auto getDouble = [&](const char* key) -> double {
            if (!vp.contains(key) || !vp[key].is_number()) throw std::runtime_error(std::string("Missing/invalid numeric key: videoParameters.") + key);
            return vp[key].get<double>();
        };
        auto getBool = [&](const char* key) -> bool {
            if (!vp.contains(key) || !vp[key].is_boolean()) throw std::runtime_error(std::string("Missing/invalid boolean key: videoParameters.") + key);
            return vp[key].get<bool>();
        };
        auto getString = [&](const char* key) -> std::string {
            if (!vp.contains(key) || !vp[key].is_string()) throw std::runtime_error(std::string("Missing/invalid string key: videoParameters.") + key);
            return vp[key].get<std::string>();
        };

        metadata.videoParameters.activeVideoStart = getInt("activeVideoStart");
        metadata.videoParameters.activeVideoEnd = getInt("activeVideoEnd");
        metadata.videoParameters.black16bIre = getInt("black16bIre");
        metadata.videoParameters.white16bIre = getInt("white16bIre");
        metadata.videoParameters.colourBurstStart = getInt("colourBurstStart");
        metadata.videoParameters.colourBurstEnd = getInt("colourBurstEnd");
        metadata.videoParameters.fieldWidth = getInt("fieldWidth");
        metadata.videoParameters.fieldHeight = getInt("fieldHeight");
        metadata.videoParameters.system = getString("system");
        metadata.videoParameters.sampleRate = getDouble("sampleRate");
        if (vp.contains("chromaGain")) {
            if (!vp["chromaGain"].is_number()) throw std::runtime_error("Missing/invalid numeric key: videoParameters.chromaGain");
            metadata.videoParameters.chromaGain = vp["chromaGain"].get<double>();
        }
        metadata.videoParameters.isWidescreen = getBool("isWidescreen");

        metadata.fields.clear();
        if (!root.contains("fields") || !root["fields"].is_array()) {
            error = "Metadata JSON is missing required array: fields";
            return false;
        }
        for (const json& item : root["fields"]) {
            if (!item.is_object()) continue;
            if (!item.contains("fieldPhaseID") || !item["fieldPhaseID"].is_number_integer()) continue;
            LdJsonFieldMeta fieldMeta;
            fieldMeta.fieldPhaseID = item["fieldPhaseID"].get<int>();
            fieldMeta.isFirstField = item.contains("isFirstField") && item["isFirstField"].is_boolean() ? item["isFirstField"].get<bool>() : false;
            fieldMeta.seqNo = item.contains("seqNo") && item["seqNo"].is_number_integer() ? item["seqNo"].get<int>() : static_cast<int>(metadata.fields.size());
            metadata.fields.push_back(fieldMeta);
        }

        if (metadata.fields.empty()) {
            error = "Metadata JSON contains no usable field phase entries in fields[]";
            return false;
        }
        std::sort(metadata.fields.begin(), metadata.fields.end(), [](const LdJsonFieldMeta& a, const LdJsonFieldMeta& b) { return a.seqNo < b.seqNo; });
        return true;
    } catch (const std::exception& e) {
        error = std::string("Failed to parse metadata JSON: ") + e.what();
        return false;
    }
}

std::string Y4mNtscWriter::normalizeSystem(const std::string& system) {
    std::string normalized;
    normalized.reserve(system.size());
    for (char c : system) {
        if (!std::isspace(static_cast<unsigned char>(c))) normalized.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return normalized;
}

std::size_t Y4mNtscWriter::inferCyclePeriod(const std::vector<int>& phases) {
    if (phases.empty()) return 0;
    const std::size_t maxPeriod = std::min<std::size_t>(32, phases.size());
    for (std::size_t period = 1; period <= maxPeriod; ++period) {
        bool cycleMatches = true;
        for (std::size_t i = 0; i < phases.size(); ++i) {
            if (phases[i] != phases[i % period]) {
                cycleMatches = false;
                break;
            }
        }
        if (cycleMatches) return period;
    }
    return phases.size();
}

bool Y4mNtscWriter::getLinePhase(int lineNumber, int firstFieldPhaseID, int secondFieldPhaseID) {
    const int fieldID = (lineNumber % 2 == 0) ? firstFieldPhaseID : secondFieldPhaseID;
    const bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);
    const int fieldLine = lineNumber / 2;
    const bool isEvenLine = (fieldLine % 2) == 0;
    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}

uint16_t Y4mNtscWriter::mapYToLimited(double y) const {
    const double minValue = config.forceLimited ? Y_LEGAL_MIN : Y_MIN;
    const double maxValue = config.forceLimited ? Y_LEGAL_MAX : Y_MAX;
    return static_cast<uint16_t>(std::clamp(((y - yOffset) * yScale) + Y_ZERO, minValue, maxValue));
}

uint16_t Y4mNtscWriter::mapUToLimited(double u) const {
    const double minValue = config.forceLimited ? C_LEGAL_MIN : C_MIN;
    const double maxValue = config.forceLimited ? C_LEGAL_MAX : C_MAX;
    return static_cast<uint16_t>(std::clamp((u * cbScale) + C_ZERO, minValue, maxValue));
}

uint16_t Y4mNtscWriter::mapVToLimited(double v) const {
    const double minValue = config.forceLimited ? C_LEGAL_MIN : C_MIN;
    const double maxValue = config.forceLimited ? C_LEGAL_MAX : C_MAX;
    return static_cast<uint16_t>(std::clamp((v * crScale) + C_ZERO, minValue, maxValue));
}

double Y4mNtscWriter::convolve5Tap(const std::vector<double>& data, int index) const {
    static constexpr double taps[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
    const int last = static_cast<int>(data.size()) - 1;
    double sum = 0.0;
    for (int i = -2; i <= 2; ++i) {
        const int x = std::clamp(index + i, 0, last);
        sum += data[x] * taps[i + 2];
    }
    return sum;
}

Y4mNtscWriter::Y4mNtscWriter(const LdJsonMetadata& metadata_, const Y4mNtscConfig& config_, std::ostream& output_)
    : metadata(metadata_), config(config_), output(&output_) {
    const std::string system = normalizeSystem(metadata.videoParameters.system);
    if (system != "NTSC" && system != "NTSC-J") throw std::runtime_error("Y4M mode supports NTSC/NTSC-J metadata only.");
    frameWidth = metadata.videoParameters.fieldWidth;
    frameHeight = (metadata.videoParameters.fieldHeight * 2) - 1;
    if (frameWidth <= 0 || frameHeight <= 0) throw std::runtime_error("Metadata contains invalid frame dimensions.");
    const int black16bIre = config.levelsOverride ? config.black16bIreOverride : metadata.videoParameters.black16bIre;
    const int white16bIre = config.levelsOverride ? config.white16bIreOverride : metadata.videoParameters.white16bIre;
    if (white16bIre <= black16bIre) throw std::runtime_error("Resolved black/white levels are invalid.");
    if (config.chromaGain < 0.0 || !std::isfinite(config.chromaGain)) throw std::runtime_error("Resolved chroma gain is invalid.");

    xStart = 0;
    xEnd = frameWidth;
    yStart = 0;
    yEnd = frameHeight;
    if (!config.fullFrame) {
        xStart = (config.activeVideoStartOverride >= 0) ? config.activeVideoStartOverride : metadata.videoParameters.activeVideoStart;
        xEnd = (config.activeVideoEndOverride >= 0) ? config.activeVideoEndOverride : metadata.videoParameters.activeVideoEnd;
        yStart = config.firstLine;
        yEnd = config.lastLine;
    }
    if (xStart < 0 || xEnd > frameWidth || xStart >= xEnd) throw std::runtime_error("Invalid horizontal Y4M output range.");
    if (yStart < 0 || yEnd > frameHeight || yStart >= yEnd) throw std::runtime_error("Invalid vertical Y4M output range.");
    outputWidth = xEnd - xStart;
    outputHeight = yEnd - yStart;

    yOffset = static_cast<double>(black16bIre);
    yRange = static_cast<double>(white16bIre - black16bIre);
    yScale = Y_SCALE / yRange;
    cbScale = (C_SCALE / (ONE_MINUS_Kb * kB)) / yRange;
    crScale = (C_SCALE / (ONE_MINUS_Kr * kR)) / yRange;

    const double theta = (ROTATE_DEGREES * PI_D) / 180.0;
    rotateSin = std::sin(theta);
    rotateCos = std::cos(theta);

    std::vector<int> phases;
    phases.reserve(metadata.fields.size());
    for (const LdJsonFieldMeta& field : metadata.fields) {
        if (field.fieldPhaseID >= 1 && field.fieldPhaseID <= 4) phases.push_back(field.fieldPhaseID);
    }
    if (phases.empty()) throw std::runtime_error("Metadata fields[] does not contain valid fieldPhaseID values in range 1..4.");
    const std::size_t period = inferCyclePeriod(phases);
    phaseCycle.assign(phases.begin(), phases.begin() + static_cast<std::ptrdiff_t>(period));
    frameIndex = config.frameIndexOffset;

    const char interlaceTag = metadata.fields.front().isFirstField ? 't' : 'b';
    const char* aspectTag = metadata.videoParameters.isWidescreen ? "25:22" : "352:413";

    std::ostringstream header;
    header << "YUV4MPEG2 W" << outputWidth << " H" << outputHeight << " F30000:1001 I" << interlaceTag << " A" << aspectTag << " C444p16 XCOLORRANGE=LIMITED\n";
    (*output) << header.str();
    if (!(*output)) throw std::runtime_error("Failed while writing Y4M stream header.");
}

void Y4mNtscWriter::writeFrame(const std::vector<uint16_t>& lumaPlane, const std::vector<uint16_t>& chromaPlane) {
    const std::size_t requiredSamples = static_cast<std::size_t>(frameWidth) * static_cast<std::size_t>(frameHeight);
    if (lumaPlane.size() < requiredSamples || chromaPlane.size() < requiredSamples) throw std::runtime_error("Y4M writer received frame planes smaller than required by metadata dimensions.");

    const std::size_t phaseBase = frameIndex * 2;
    const int firstFieldPhaseID = phaseCycle[phaseBase % phaseCycle.size()];
    const int secondFieldPhaseID = phaseCycle[(phaseBase + 1) % phaseCycle.size()];
    frameIndex++;

    const std::size_t framePixels = static_cast<std::size_t>(outputWidth) * static_cast<std::size_t>(outputHeight);
    std::vector<uint16_t> yPlane(framePixels);
    std::vector<uint16_t> cbPlane(framePixels);
    std::vector<uint16_t> crPlane(framePixels);
    std::vector<double> lineI(static_cast<std::size_t>(outputWidth));
    std::vector<double> lineQ(static_cast<std::size_t>(outputWidth));

    for (int outY = 0; outY < outputHeight; ++outY) {
        const int srcY = yStart + outY;
        const bool linePhase = getLinePhase(srcY, firstFieldPhaseID, secondFieldPhaseID);
        double si = 0.0;
        double sq = 0.0;

        for (int outX = 0; outX < outputWidth; ++outX) {
            const int srcX = xStart + outX;
            const std::size_t srcIndex = static_cast<std::size_t>(srcY) * static_cast<std::size_t>(frameWidth) + static_cast<std::size_t>(srcX);
            double cavg = static_cast<double>(chromaPlane[srcIndex]) - 32768.0;
            if (linePhase) cavg = -cavg;
            switch (srcX % 4) {
                case 0: sq = cavg; break;
                case 1: si = -cavg; break;
                case 2: sq = -cavg; break;
                case 3: si = cavg; break;
                default: break;
            }

            lineI[static_cast<std::size_t>(outX)] = si;
            lineQ[static_cast<std::size_t>(outX)] = sq;

            const std::size_t outIndex = static_cast<std::size_t>(outY) * static_cast<std::size_t>(outputWidth) + static_cast<std::size_t>(outX);
            yPlane[outIndex] = mapYToLimited(static_cast<double>(lumaPlane[srcIndex]));
        }

        for (int outX = 0; outX < outputWidth; ++outX) {
            const double filteredI = convolve5Tap(lineI, outX);
            const double filteredQ = convolve5Tap(lineQ, outX);
            const double u = config.chromaGain * ((-rotateSin * filteredI) + (rotateCos * filteredQ));
            const double v = config.chromaGain * ((rotateCos * filteredI) + (rotateSin * filteredQ));
            const std::size_t outIndex = static_cast<std::size_t>(outY) * static_cast<std::size_t>(outputWidth) + static_cast<std::size_t>(outX);
            cbPlane[outIndex] = mapUToLimited(u);
            crPlane[outIndex] = mapVToLimited(v);
        }
    }

    output->write("FRAME\n", 6);
    output->write(reinterpret_cast<const char*>(yPlane.data()), static_cast<std::streamsize>(yPlane.size() * sizeof(uint16_t)));
    output->write(reinterpret_cast<const char*>(cbPlane.data()), static_cast<std::streamsize>(cbPlane.size() * sizeof(uint16_t)));
    output->write(reinterpret_cast<const char*>(crPlane.data()), static_cast<std::streamsize>(crPlane.size() * sizeof(uint16_t)));
    if (!(*output)) throw std::runtime_error("Failed while writing Y4M frame payload.");
}
