using System;
using System.Collections.Generic;
using System.Text;

namespace RapidOCRSharpOnnx.Test
{
    public class UnitTestBase
    {
        protected const string detectModelName = "ch_PP-OCRv5_det_mobile.onnx";
        protected const string clsModelMobileName = "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx";
        protected const string clsModelServerName = "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx";
        protected const string recModelName = "ch_PP-OCRv5_rec_mobile.onnx";
        protected const string txtpng = "txt.png";
        protected const string headerTextpng = "headerText.png";
        protected const string testClspng = "test_cls.png";

        protected const string txtRes = "Let's start collaborating Select a chat or channel from the list to get started.";
        protected const string headerTextRes = "Project=378 Dragonfly Printer=139 DRA-MP-20P5 Page=45285 (1/1) Pattern=HPGHOST_A4.pclm Crg_TK [DRA-MP-20P5-35] Page=5 Formatter/Alt Firmware [DA0iCA2513BR] Page=7620 Engine DCc Firmware [00000] Page=45285 Date=04/04/25 Time=09:13:11 Input Tray=Tray 1 Test Name=PQsuite Dragonfly_Mp Plex Mode=Simplex Media=3051 A4-80g-Blue M&G Multipurpose Paper Temp=15.0 Humidity=10.0 Fuser Mode-Plain";

        protected string detectPath;
        protected string clsMobilePath;
        protected string clsServerPath;
        protected string recPath;

        public UnitTestBase()
        {
            detectPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", detectModelName);
            clsMobilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelMobileName);
            clsServerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", clsModelServerName);
            recPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", recModelName);
        }

        protected string GetFullPath(string img)
        {
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages", img);
        }

    }
}
