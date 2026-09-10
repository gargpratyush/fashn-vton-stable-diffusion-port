# Report imagery, provenance and licensing

The main repository's code license must not be applied indiscriminately
to model weights, external SDKs or report imagery.

## Photographic fixtures

The cardigan, dress, trousers and crochet-top studies use public
[Zheng-Chong/CatVTON demo assets](https://github.com/Zheng-Chong/CatVTON/tree/999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e)
at revision `999bdbe81e6008a3f5749af7c1e0b0fa3d21b48e`.
The source repository states
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Preserve attribution, noncommercial and share-alike requirements when
redistributing that imagery or adaptations. Report crops, contact sheets,
generated try-on adaptations and amplified differences are identified in
their captions and manifests; they are not unchanged source photographs.

Original FASHN person/worn-garment examples are from
[FASHN AI / fashn-vton-1.5](https://github.com/fashn-AI/fashn-vton-1.5/tree/7c0f10af3f91ad4048fe9729c470a13ef905d25a),
revision `7c0f10af3f91ad4048fe9729c470a13ef905d25a`.
Consult that project's Apache-2.0 license and any applicable source-image
rights; this report is not a representation that all depicted people or
third-party designs are licensed for arbitrary commercial use.

The initial SD 1.5 smoke result is a separately labeled generated
diagnostic, not a FASHN photographic-quality sample.

## Models and dependencies

FASHN model/source, DWPose, YOLOX, MMPose, OpenCV and ONNX Runtime have their
own licenses and notices. See the native preparer's
[NOTICE](../examples/fashn-preprocess/NOTICE.txt).
The upstream FASHN source license is preserved at
[licenses/FASHN-Apache-2.0.txt](licenses/FASHN-Apache-2.0.txt).

The optional FASHN human parser inherits NVIDIA SegFormer's separate
**noncommercial research/evaluation-only** license. The local experiment
authorization was explicitly limited to research/evaluation; it is not a
commercial license grant. Parser weights and binaries are not distributed
here. Use requires separate provisioning and explicit acknowledgement.

No claim of photorealistic identity preservation, garment-construction
fidelity, legal clearance for deployment or general product certification
is made by the numerical comparisons.
