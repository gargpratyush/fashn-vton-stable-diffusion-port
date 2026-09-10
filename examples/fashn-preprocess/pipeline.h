#ifndef __SD_FASHN_PIPELINE_H__
#define __SD_FASHN_PIPELINE_H__

#include <functional>
#include <map>
#include <stdexcept>
#include "preprocess.h"

namespace fashn_prepare {

    struct PreparationCancelled : std::runtime_error {
        PreparationCancelled()
            : std::runtime_error("Native preparation cancelled") {}
    };

    struct PreparedImages {
        std::array<cv::Mat, 4> images;
        cv::Rect crop;
        cv::Mat person_preresize, garment_preresize;
        cv::Mat person_segmentation, garment_segmentation;
        Pose person_points{}, garment_points{};
    };

    const std::map<std::string, std::string>& pose_weight_hashes();

    // Reuses local CPU sessions; callers must serialize prepare() calls.
    class Preparer {
        CPUOptions cpu_;
        Model detector_;
        Model pose_;
        std::unique_ptr<Model> parser_;
        std::string parser_manifest_ = "null";
        std::filesystem::path parser_directory_;
        Model& parser();

    public:
        Preparer(const std::filesystem::path& pose_dir,
                 const std::filesystem::path& parser_dir = {},
                 bool accept_parser_research_license     = false,
                 bool lazy_parser                       = true,
                 const CPUOptions& cpu                  = {});
        bool has_parser() const { return !parser_directory_.empty(); }
        bool parser_loaded() const { return parser_ != nullptr; }
        const std::string& parser_manifest() const { return parser_manifest_; }
        PreparedImages prepare(const cv::Mat& person_bgr, const cv::Mat& garment_bgr, const std::string& category, const std::string& photo_type, bool segmentation_free, const std::function<bool()>& cancelled = {});
    };

}  // namespace fashn_prepare
#endif
